"""Bounded, OpenAI-compatible chat-completions client using the standard library."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

from . import __version__
from .grading import strict_json_loads

SYSTEM_PROMPT = "Return only the answer. No explanation."
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_RETRY_WAIT_SECONDS = 60.0


class APIError(RuntimeError):
    def __init__(
        self, message: str, *, kind: str = "api_error", fatal: bool = False,
        http_attempts: list[dict[str, Any]] | None = None, latency_ms: int = 0,
        unknown_usage: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.fatal = fatal
        self.http_attempts = list(http_attempts or [])
        self.latency_ms = latency_ms
        self.unknown_usage = unknown_usage


@dataclass(frozen=True)
class APIResponse:
    text: str
    usage: dict[str, Any]
    latency_ms: int
    response_id: str | None = None
    finish_reason: str | None = "stop"
    refusal: str | None = None
    http_attempts: list[dict[str, Any]] = field(default_factory=list)
    unknown_usage: bool = False
    response_model: str | None = None
    system_fingerprint: str | None = None


def _completion_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base URL must be a non-empty string")
    base = base_url
    if any(character.isspace() or ord(character) < 32 for character in base):
        raise ValueError("base URL must not contain whitespace or control characters")
    try:
        parsed = urllib.parse.urlsplit(base)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid base URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain credentials, a query, or a fragment")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port != (443 if parsed.scheme == "https" else 80):
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urllib.parse.urlunsplit((parsed.scheme, host, path, "", ""))


def _protocol_error(message: str) -> APIError:
    return APIError(message, kind="protocol_error", unknown_usage=True)


def _optional_string(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise _protocol_error(f"response {name} must be a string or null")
    return value


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _protocol_error("response message.content must be a string, list, or null")
    parts: list[str] = []
    for item in content:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("type"), str)
            or item["type"] not in {"text", "output_text"}
        ):
            raise _protocol_error("response message.content contains an unsupported part")
        text = item.get("text")
        if not isinstance(text, str):
            raise _protocol_error("response content text must be a string")
        parts.append(text)
    return "".join(parts)


def _usage(value: Any, depth: int = 0) -> dict[str, Any]:
    if not isinstance(value, dict) or depth > 8:
        raise _protocol_error("response usage must be an object with bounded nesting")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, dict):
            result[key] = _usage(item, depth + 1)
        elif isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
        else:
            raise _protocol_error("response usage counters must be non-negative integers")
    return result


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive_number(value: Any, name: str) -> None:
    if not _finite_number(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatCompletionsClient:
    def __init__(
        self, *, base_url: str, api_key: str | None, model: str,
        timeout: float = 60.0, retries: int = 2, temperature: float | None = 0.0,
        seed: int | None = None, token_param: str = "max_tokens",
        max_requests: int | None = None, deadline: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        endpoint = _completion_url(base_url)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if api_key is not None and (
            not isinstance(api_key, str) or any(ord(c) < 32 or ord(c) == 127 for c in api_key)
        ):
            raise ValueError("API key must be a string without control characters")
        if api_key is not None:
            try:
                api_key.encode("latin-1")
            except UnicodeEncodeError:
                raise ValueError("API key must be encodable as an HTTP header") from None
        _positive_number(timeout, "timeout")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        if temperature is not None and (
            not _finite_number(temperature) or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be a finite number in [0, 2]")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer")
        if not isinstance(token_param, str) or token_param not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("unsupported token parameter")
        if max_requests is not None and (
            isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1
        ):
            raise ValueError("max_requests must be a positive integer")
        if deadline is not None:
            _positive_number(deadline, "deadline")
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
        ):
            raise ValueError("reasoning_effort must be a non-empty string")
        self.base_url = base_url
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model.strip()
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.seed = seed
        self.token_param = token_param
        self.max_requests = max_requests
        self.deadline = deadline
        self.reasoning_effort = reasoning_effort
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._http_requests = 0
        self._unknown_usage_requests = 0

    def configuration(self) -> dict[str, Any]:
        return {
            "surface": "chat_completions", "endpoint": self.endpoint, "model": self.model,
            "system_prompt_digest": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "temperature": self.temperature, "seed": self.seed, "token_param": self.token_param,
            "reasoning_effort": self.reasoning_effort,
        }

    def budget_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "http_requests": self._http_requests, "max_requests": self.max_requests,
                "total_elapsed_ms": self._elapsed(self._started), "deadline_seconds": self.deadline,
                "unknown_usage": self._unknown_usage_requests > 0,
                "unknown_usage_requests": self._unknown_usage_requests,
            }

    @staticmethod
    def _elapsed(started: float) -> int:
        return max(0, round((time.monotonic() - started) * 1000))

    def _remaining(self) -> float | None:
        if self.deadline is None:
            return None
        remaining = self.deadline - (time.monotonic() - self._started)
        if remaining <= 0:
            raise APIError("run deadline exhausted", kind="budget_exhausted", fatal=True)
        return remaining

    def _check_budget(self) -> float:
        # The caller holds the lock when checking and reserving a request slot.
        if self.max_requests is not None and self._http_requests >= self.max_requests:
            raise APIError("HTTP request budget exhausted", kind="budget_exhausted", fatal=True)
        remaining = self._remaining()
        return self.timeout if remaining is None else min(self.timeout, remaining)

    def _reserve_request(self) -> tuple[int, float]:
        with self._lock:
            timeout = self._check_budget()
            self._http_requests += 1
            return self._http_requests, timeout

    def _open(self, request: urllib.request.Request, timeout: float):
        # Each request owns its handlers; workers share no mutable opener state.
        return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)

    def _read_body(self, response: Any, limit: int, *, truncate: bool = False) -> bytes:
        chunks: list[bytes] = []
        size = 0
        read = getattr(response, "read1", response.read)
        while size < limit + (not truncate):
            remaining = self._remaining()
            if remaining is not None:
                sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                if sock is not None:
                    sock.settimeout(min(self.timeout, remaining))
            chunk = read(min(65536, limit + (not truncate) - size))
            self._remaining()
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > limit:
            raise _protocol_error(f"response exceeds the {limit}-byte limit")
        return b"".join(chunks)

    def _redact(self, value: Any) -> str:
        text = str(value)
        if self.api_key:
            text = text.replace(self.api_key, "[REDACTED]")
        return text

    @staticmethod
    def _retry_after(value: str | None) -> float | None:
        if not value:
            return None
        try:
            seconds = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = date.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                return None
        return max(0.0, seconds) if math.isfinite(seconds) else None

    def _wait_retry(self, delay: float) -> None:
        with self._lock:
            self._check_budget()
            remaining = self._remaining()
            if remaining is not None and delay >= remaining:
                raise APIError("retry would exceed run deadline", kind="budget_exhausted", fatal=True)
            if delay > MAX_RETRY_WAIT_SECONDS:
                # Stop instead of capping the delay and retrying before Retry-After.
                raise APIError(
                    "server requested a retry wait above 60 seconds; retry deferred",
                    kind="retry_deferred", fatal=True,
                )
        time.sleep(delay)

    def _parse_response(self, raw: bytes) -> dict[str, Any]:
        try:
            data = strict_json_loads(raw.decode("utf-8"), path="response")
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise _protocol_error("response is not valid UTF-8 JSON") from exc
        if not isinstance(data, dict):
            raise _protocol_error("response root must be an object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise _protocol_error("response must contain a non-empty choices list of objects")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise _protocol_error("response choice.message must be an object")
        refusal = _optional_string(message.get("refusal"), "message.refusal")
        if "content" not in message and refusal is None:
            raise _protocol_error("response message has neither content nor refusal")
        usage = _usage(data["usage"]) if data.get("usage") is not None else {}
        return {
            "text": _content_to_text(message.get("content")), "usage": usage,
            "response_id": _optional_string(data.get("id"), "id"),
            "finish_reason": _optional_string(choices[0].get("finish_reason"), "finish_reason"),
            "refusal": refusal,
            "response_model": _optional_string(data.get("model"), "model"),
            "system_fingerprint": _optional_string(data.get("system_fingerprint"), "system_fingerprint"),
        }

    def complete(self, prompt: str, max_tokens: int) -> APIResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        started = time.monotonic()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            self.token_param: max_tokens,
        }
        for key in ("temperature", "seed", "reasoning_effort"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": f"model-sentinel/{__version__}"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        attempts: list[dict[str, Any]] = []

        def enriched(error: APIError) -> APIError:
            return APIError(
                self._redact(error), kind=error.kind, fatal=error.fatal,
                http_attempts=attempts, latency_ms=self._elapsed(started),
                unknown_usage=error.unknown_usage or any(item["unknown_usage"] for item in attempts),
            )

        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            try:
                request_index, timeout = self._reserve_request()
            except APIError as exc:
                raise enriched(exc) from None
            attempt_started = time.monotonic()
            status_code: int | None = None
            retry_after: float | None = None
            retryable = False
            result: dict[str, Any] | None = None
            error: APIError | None = None
            try:
                with self._open(request, timeout) as response:
                    status_code = getattr(response, "status", 200)
                    raw = self._read_body(response, MAX_RESPONSE_BYTES)
                result = self._parse_response(raw)
                self._remaining()
            except urllib.error.HTTPError as exc:
                status_code = exc.code
                retryable = exc.code in {408, 409, 425, 429} or 500 <= exc.code < 600
                retry_after = self._retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                try:
                    detail = self._read_body(exc, 2048, truncate=True).decode("utf-8", errors="replace")
                except (APIError, OSError, http.client.HTTPException):
                    detail = "unable to read error response"
                finally:
                    exc.close()
                kind = "redirect" if 300 <= exc.code < 400 else "http_error"
                error = APIError(
                    self._redact(f"HTTP {exc.code}: {detail}"), kind=kind,
                    fatal=not retryable, unknown_usage=exc.code >= 500 or exc.code == 408,
                )
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                retryable = True
                error = APIError(self._redact(exc), kind="transport_error", unknown_usage=True)
            except APIError as exc:
                error = exc
                result = None
                if exc.kind == "budget_exhausted":
                    error.unknown_usage = True
            if error is not None and error.kind != "budget_exhausted":
                try:
                    self._remaining()
                except APIError as exc:
                    exc.unknown_usage = error.unknown_usage
                    error = exc
                    retryable = False
            record = {
                "attempt": attempt + 1, "request_index": request_index,
                "status_code": status_code, "latency_ms": self._elapsed(attempt_started),
                "error_kind": error.kind if error else None,
                "unknown_usage": error.unknown_usage if error else not bool(result and result["usage"]),
                "usage": result["usage"] if result else {},
            }
            attempts.append(record)
            if record["unknown_usage"]:
                with self._lock:
                    self._unknown_usage_requests += 1
            if result is not None:
                return APIResponse(
                    **result, latency_ms=self._elapsed(started), http_attempts=attempts,
                    unknown_usage=any(item["unknown_usage"] for item in attempts),
                )
            assert error is not None
            if error.fatal or not retryable or attempt >= self.retries:
                raise enriched(error) from None
            backoff = min(0.5 * 2 ** min(attempt, 6), 30.0)
            delay = max(retry_after or 0.0, random.uniform(backoff / 2, backoff))
            try:
                self._wait_retry(delay)
            except APIError as exc:
                raise enriched(exc) from None
        raise AssertionError("unreachable retry state")
