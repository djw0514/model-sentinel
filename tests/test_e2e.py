"""Exercise the actual HTTP client, CLI and saved-report round trip together."""

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from model_sentinel.cli import main
from model_sentinel.pack import load_pack


class EndToEndTests(unittest.TestCase):
    def test_local_endpoint_baseline_standard_deep_compare_and_budget(self):
        pack = load_pack()
        answers = {}
        late_prompts = {case["prompt"] for case in pack["cases"][12:]}
        for case in pack["cases"]:
            grader = case["grader"]
            answers[case["prompt"]] = (grader["answers"][0] if grader["type"] == "exact"
                                       else json.dumps(grader["value"]))
        state = {"regressed": False, "requests": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = request["messages"][-1]["content"]
                with lock:
                    state["requests"] += 1
                    content = "wrong" if state["regressed"] and prompt in late_prompts else answers[prompt]
                response = {
                    "id": "local-response", "model": "local-test",
                    "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                              "completion_tokens_details": {"reasoning_tokens": 1}},
                }
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "SENTINEL_TEST_KEY": "synthetic-local-key", "NO_PROXY": "127.0.0.1",
            }):
                root = Path(directory)
                common = ["--model", "local-test", "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                          "--api-key-env", "SENTINEL_TEST_KEY", "--timeout", "5", "--retries", "0"]

                def invoke(arguments):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        return main(arguments)

                baseline = root / "baseline.json"
                self.assertEqual(invoke(["run", *common, "--repeats", "3", "--workers", "4",
                                         "--output", str(baseline)]), 0)
                check = root / "check.json"
                self.assertEqual(invoke(["check", *common, "--baseline", str(baseline),
                                         "--sample-seed", "42", "--output", str(check)]), 0)
                report = json.loads(check.read_text(encoding="utf-8"))
                self.assertEqual(report["verdict"]["samples"], 12)
                self.assertEqual(report["budget"]["http_requests"], 12)
                self.assertEqual(report["usage"]["completion_tokens_details"]["reasoning_tokens"], 12)

                state["regressed"] = True
                deep = root / "deep.json"
                self.assertEqual(invoke(["check", *common, "--mode", "deep", "--workers", "4",
                                         "--baseline", str(baseline), "--output", str(deep)]), 2)
                deep_report = json.loads(deep.read_text(encoding="utf-8"))
                self.assertEqual(deep_report["verdict"]["confirmed_regressions"], 20)
                candidate = root / "candidate.json"
                candidate.write_text(json.dumps(deep_report["candidate_run"]), encoding="utf-8")
                self.assertEqual(invoke(["compare", str(baseline), str(candidate)]), 2)

                state["regressed"] = False
                before = state["requests"]
                partial = root / "partial.json"
                self.assertEqual(invoke(["check", *common, "--baseline", str(baseline),
                                         "--max-requests", "1", "--output", str(partial)]), 1)
                partial_report = json.loads(partial.read_text(encoding="utf-8"))
                self.assertTrue(partial_report["aborted"])
                self.assertEqual(partial_report["verdict"]["decision"], "inconclusive")
                self.assertEqual(state["requests"] - before, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
