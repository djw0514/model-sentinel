import io
import json
import threading
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from model_sentinel.api import APIError, ChatCompletionsClient


def response_data(**overrides):
    data = {
        "id": "response-1",
        "model": "snapshot-1",
        "system_fingerprint": "fp-1",
        "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }
    data.update(overrides)
    return data


def response(data=None):
    return io.BytesIO(json.dumps(response_data() if data is None else data).encode())


def client(**overrides):
    args = {"base_url": "https://example.invalid/v1", "api_key": "test-secret", "model": "test", "retries": 0}
    args.update(overrides)
    return ChatCompletionsClient(**args)


def http_error(code, *, body=b"error", retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://example.invalid", code, "error", headers, io.BytesIO(body))


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class APIValidationTests(unittest.TestCase):
    def test_invalid_configuration_fails_before_network(self):
        invalid = [
            {"base_url": "file:///tmp/data"}, {"base_url": "https://user:pass@example.invalid"},
            {"base_url": "https://example.invalid?key=secret"}, {"base_url": "https://example.invalid/#key"},
            {"base_url": "https://example.invalid\n"}, {"base_url": "https://example.invalid:bad"},
            {"timeout": 0}, {"timeout": -1}, {"timeout": float("inf")}, {"timeout": True},
            {"timeout": 10 ** 400}, {"temperature": 10 ** 400}, {"deadline": 10 ** 400},
            {"retries": -1}, {"retries": True}, {"retries": 1.5},
            {"temperature": float("nan")}, {"temperature": -0.1}, {"temperature": 3},
            {"seed": True}, {"seed": 1.5}, {"model": " "}, {"model": 1},
            {"max_requests": 0}, {"max_requests": -1}, {"deadline": 0},
            {"deadline": float("nan")}, {"reasoning_effort": ""},
            {"api_key": "bad\nkey"}, {"api_key": "\u6d4b\u8bd5"}, {"token_param": "tokens"}, {"token_param": []},
        ]
        with patch.object(ChatCompletionsClient, "_open") as opened:
            for config in invalid:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    client(**config)
            opened.assert_not_called()

    def test_invalid_call_does_not_consume_budget(self):
        c = client(max_requests=1)
        for prompt, cap in [(None, 8), ("", 8), ("x", 0), ("x", -1), ("x", True), ("x", 1.5)]:
            with self.subTest(prompt=prompt, cap=cap), self.assertRaises(ValueError):
                c.complete(prompt, cap)
        self.assertEqual(c.budget_snapshot()["http_requests"], 0)

    def test_configuration_is_canonical_and_has_no_key(self):
        c = client(base_url="HTTPS://EXAMPLE.INVALID:443/v1/chat/completions/", seed=7, reasoning_effort="low")
        config = c.configuration()
        self.assertEqual(config["endpoint"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(config["surface"], "chat_completions")
        self.assertEqual(len(config["system_prompt_digest"]), 64)
        self.assertEqual(config["seed"], 7)
        self.assertEqual(config["reasoning_effort"], "low")
        self.assertNotIn("test-secret", json.dumps(config))
        self.assertEqual(config, client(seed=7, reasoning_effort="low").configuration())

    def test_response_metadata_usage_and_payload(self):
        c = client(temperature=None, seed=3, reasoning_effort="low", token_param="max_completion_tokens")
        with patch.object(c, "_open", return_value=response()) as opened:
            result = c.complete("Compute.", 16)
        payload = json.loads(opened.call_args.args[0].data)
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["max_completion_tokens"], 16)
        self.assertEqual(payload["seed"], 3)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(result.response_model, "snapshot-1")
        self.assertEqual(result.system_fingerprint, "fp-1")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage["completion_tokens_details"]["reasoning_tokens"], 1)
        self.assertEqual(len(result.http_attempts), 1)
        self.assertFalse(result.unknown_usage)

    def test_truncation_and_refusal_are_preserved(self):
        for finish, refusal in [("length", None), ("content_filter", "refused")]:
            with self.subTest(finish=finish):
                data = response_data(choices=[{"message": {"content": None, "refusal": refusal}, "finish_reason": finish}])
                c = client()
                with patch.object(c, "_open", return_value=response(data)):
                    result = c.complete("x", 1)
                self.assertEqual(result.text, "")
                self.assertEqual(result.finish_reason, finish)
                self.assertEqual(result.refusal, refusal)

    def test_malformed_response_is_a_protocol_error(self):
        malformed = [
            [], None, {"choices": []}, {"choices": [1]}, {"choices": [{"message": None}]},
            response_data(usage=[1]), response_data(usage={"total_tokens": True}),
            response_data(usage={"total_tokens": -1}), response_data(usage={"total_tokens": 1.5}),
            response_data(choices=[{"message": {"content": {"text": "42"}}}]),
            response_data(choices=[{"message": {"content": [{"type": "text", "text": 42}]}}]),
            response_data(choices=[{"message": {"content": [{"type": [], "text": "42"}]}}]),
            response_data(choices=[{"message": {"content": "42"}, "finish_reason": 1}]),
            response_data(choices=[{"message": {"refusal": 1}}]), response_data(model=1),
        ]
        for data in malformed:
            with self.subTest(data=data):
                c = client(retries=2)
                raw = io.BytesIO(json.dumps(data).encode())
                with patch.object(c, "_open", return_value=raw) as opened, self.assertRaises(APIError) as raised:
                    c.complete("x", 8)
                self.assertEqual(raised.exception.kind, "protocol_error")
                self.assertTrue(raised.exception.unknown_usage)
                self.assertEqual(opened.call_count, 1)
                self.assertEqual(len(raised.exception.http_attempts), 1)

    def test_invalid_encoding_json_and_oversize_are_bounded(self):
        for raw in [b"\xff", b"not json", b"x" * 65]:
            with self.subTest(raw=raw[:8]):
                c = client()
                with patch.object(c, "_open", return_value=io.BytesIO(raw)), patch("model_sentinel.api.MAX_RESPONSE_BYTES", 64):
                    with self.assertRaises(APIError) as raised:
                        c.complete("x", 8)
                self.assertEqual(raised.exception.kind, "protocol_error")

    def test_missing_usage_is_marked_unknown(self):
        c = client()
        with patch.object(c, "_open", return_value=response(response_data(usage=None))):
            result = c.complete("x", 8)
        self.assertEqual(result.usage, {})
        self.assertTrue(result.unknown_usage)
        self.assertEqual(c.budget_snapshot()["unknown_usage_requests"], 1)

    def test_duplicate_fields_and_nonfinite_json_are_protocol_errors(self):
        malformed = [
            b'{"choices":[{"message":{"content":"wrong","content":"right"},"finish_reason":"stop"}]}',
            b'{"choices":[{"message":{"content":"42"},"finish_reason":"stop"}],"extra":NaN}',
            b'{"choices":[{"message":{"content":"42"},"finish_reason":"stop"}],"extra":1e400}',
        ]
        for raw in malformed:
            with self.subTest(raw=raw):
                c = client()
                with patch.object(c, "_open", return_value=io.BytesIO(raw)), self.assertRaises(APIError) as raised:
                    c.complete("x", 8)
                self.assertEqual(raised.exception.kind, "protocol_error")
                self.assertTrue(raised.exception.unknown_usage)


class APIRetryAndBudgetTests(unittest.TestCase):
    def test_retry_after_and_total_latency_include_all_attempts(self):
        clock = FakeClock()
        with patch("model_sentinel.api.time.monotonic", side_effect=clock.monotonic), patch("model_sentinel.api.time.sleep", side_effect=clock.sleep), patch("model_sentinel.api.random.uniform", return_value=0.4):
            c = client(retries=1)
            def opened(request, timeout):
                clock.now += 2
                if c.budget_snapshot()["http_requests"] == 1:
                    raise http_error(429, retry_after="3")
                return response()
            with patch.object(c, "_open", side_effect=opened):
                result = c.complete("x", 8)
        self.assertEqual(clock.sleeps, [3.0])
        self.assertEqual(result.latency_ms, 7000)
        self.assertEqual([a["latency_ms"] for a in result.http_attempts], [2000, 2000])
        self.assertEqual([a["status_code"] for a in result.http_attempts], [429, 200])
        self.assertEqual(c.budget_snapshot()["http_requests"], 2)

    def test_retry_after_accepts_http_date_and_ignores_invalid_values(self):
        with patch("model_sentinel.api.time.time", return_value=0):
            self.assertEqual(ChatCompletionsClient._retry_after("Thu, 01 Jan 1970 00:00:10 GMT"), 10)
        for value in [None, "bad", "NaN", "Infinity"]:
            self.assertIsNone(ChatCompletionsClient._retry_after(value))

    def test_excessive_retry_after_stops_without_early_retry_or_sleep(self):
        for delay in ["61", "1e300"]:
            with self.subTest(delay=delay):
                c = client(retries=2)
                with patch.object(c, "_open", side_effect=http_error(429, retry_after=delay)) as opened, patch("model_sentinel.api.time.sleep") as sleep:
                    with self.assertRaises(APIError) as raised:
                        c.complete("x", 8)
                self.assertEqual(raised.exception.kind, "retry_deferred")
                self.assertTrue(raised.exception.fatal)
                self.assertEqual(len(raised.exception.http_attempts), 1)
                self.assertEqual(raised.exception.http_attempts[0]["status_code"], 429)
                self.assertEqual(opened.call_count, 1)
                sleep.assert_not_called()

    def test_fatal_errors_do_not_retry_and_redact_credentials(self):
        for code in [400, 401, 403, 404, 422]:
            with self.subTest(code=code):
                c = client(retries=2)
                with patch.object(c, "_open", side_effect=http_error(code, body=b"Invalid key test-secret")) as opened:
                    with self.assertRaises(APIError) as raised:
                        c.complete("x", 8)
                self.assertTrue(raised.exception.fatal)
                self.assertNotIn("test-secret", str(raised.exception))
                self.assertIn("[REDACTED]", str(raised.exception))
                self.assertEqual(opened.call_count, 1)

    def test_timeout_retry_preserves_unknown_usage(self):
        c = client(retries=1)
        with patch.object(c, "_open", side_effect=[TimeoutError("test-secret timeout"), response()]), patch("model_sentinel.api.time.sleep"):
            result = c.complete("x", 8)
        self.assertTrue(result.unknown_usage)
        self.assertTrue(result.http_attempts[0]["unknown_usage"])
        self.assertFalse(result.http_attempts[1]["unknown_usage"])
        self.assertEqual(c.budget_snapshot()["unknown_usage_requests"], 1)

    def test_request_budget_includes_retries_without_extra_sleep(self):
        c = client(retries=3, max_requests=1)
        with patch.object(c, "_open", side_effect=TimeoutError("timeout")) as opened, patch("model_sentinel.api.time.sleep") as sleep:
            with self.assertRaises(APIError) as raised:
                c.complete("x", 8)
        self.assertEqual(raised.exception.kind, "budget_exhausted")
        self.assertTrue(raised.exception.fatal)
        self.assertTrue(raised.exception.unknown_usage)
        self.assertEqual(len(raised.exception.http_attempts), 1)
        self.assertEqual(opened.call_count, 1)
        sleep.assert_not_called()

    def test_request_budget_is_atomic_across_workers(self):
        c = client(max_requests=3)
        start = threading.Barrier(12)
        def invoke(_):
            start.wait(timeout=5)
            try:
                return c.complete("x", 8).text
            except APIError as exc:
                return exc.kind
        with patch.object(c, "_open", side_effect=lambda *args: response()) as opened:
            with ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(invoke, range(12)))
        self.assertEqual(results.count("42"), 3)
        self.assertEqual(results.count("budget_exhausted"), 9)
        self.assertEqual(opened.call_count, 3)
        self.assertEqual(c.budget_snapshot()["http_requests"], 3)

    def test_deadline_clamps_timeout_and_prevents_long_retry_wait(self):
        clock = FakeClock()
        with patch("model_sentinel.api.time.monotonic", side_effect=clock.monotonic), patch("model_sentinel.api.time.sleep") as sleep:
            c = client(timeout=60, deadline=5, retries=2)
            clock.now += 1
            with patch.object(c, "_open", side_effect=http_error(429, retry_after="10")) as opened:
                with self.assertRaises(APIError) as raised:
                    c.complete("x", 8)
            self.assertEqual(opened.call_args.args[1], 4)
            self.assertEqual(raised.exception.kind, "budget_exhausted")
            self.assertEqual(opened.call_count, 1)
            sleep.assert_not_called()

    def test_expired_deadline_does_not_start_a_request(self):
        clock = FakeClock()
        with patch("model_sentinel.api.time.monotonic", side_effect=clock.monotonic):
            c = client(deadline=1)
            clock.now += 2
            with patch.object(c, "_open") as opened, self.assertRaises(APIError) as raised:
                c.complete("x", 8)
            self.assertEqual(raised.exception.kind, "budget_exhausted")
            self.assertEqual(c.budget_snapshot()["http_requests"], 0)
            opened.assert_not_called()

    def test_deadline_applies_while_reading_response(self):
        clock = FakeClock()
        class SlowBody(io.BytesIO):
            def read1(self, size):
                clock.now += 2
                return super().read1(size)
        with patch("model_sentinel.api.time.monotonic", side_effect=clock.monotonic):
            c = client(deadline=1)
            with patch.object(c, "_open", return_value=SlowBody(b"{}")), self.assertRaises(APIError) as raised:
                c.complete("x", 8)
            self.assertEqual(raised.exception.kind, "budget_exhausted")
            self.assertTrue(raised.exception.unknown_usage)
            self.assertEqual(len(raised.exception.http_attempts), 1)


class APIRedirectTests(unittest.TestCase):
    def test_cross_host_redirect_never_receives_authorization(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/collect")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                received.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            c = client(base_url=f"http://127.0.0.1:{server.server_port}/v1", retries=2)
            with self.assertRaises(APIError) as raised:
                c.complete("x", 8)
            self.assertEqual(raised.exception.kind, "redirect")
            self.assertTrue(raised.exception.fatal)
            self.assertEqual(c.budget_snapshot()["http_requests"], 1)
            self.assertEqual(received, [])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
