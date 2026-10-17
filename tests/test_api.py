import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from model_sentinel.api import ChatCompletionsClient


class Handler(BaseHTTPRequestHandler):
    request_data = None
    authorization = None

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        Handler.request_data = json.loads(self.rfile.read(length))
        Handler.authorization = self.headers.get("Authorization")
        body = json.dumps(
            {
                "id": "response-1",
                "choices": [{"message": {"content": "489"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class APITests(unittest.TestCase):
    def test_chat_completions_request_and_response(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = ChatCompletionsClient(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="secret",
                model="test-model",
                retries=0,
            )
            response = client.complete("Compute.", 8)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(response.text, "489")
        self.assertEqual(response.usage["total_tokens"], 13)
        self.assertEqual(Handler.request_data["model"], "test-model")
        self.assertEqual(Handler.request_data["max_tokens"], 8)
        self.assertEqual(Handler.authorization, "Bearer secret")


if __name__ == "__main__":
    unittest.main()
