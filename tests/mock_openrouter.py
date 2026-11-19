"""Tiny local stand-in for OpenRouter's POST /api/v1/images, for trying the node without a real key.

Run:    python tests/mock_openrouter.py            (listens on http://127.0.0.1:8765)
Then start ComfyUI with:
        OPENROUTER_IMAGES_URL=http://127.0.0.1:8765/api/v1/images ComfyUI
Any api_key works. Returns a solid-colour image at the requested WIDTHxHEIGHT (default 512x512);
transparent when background=transparent. A prompt containing "error" returns a 400 to test failures.
"""
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        print("request:", {k: (v if k != "input_references" else f"<{len(v)} refs>") for k, v in body.items()})
        if not self.headers.get("Authorization", "").startswith("Bearer ") or "error" in body.get("prompt", ""):
            return self.reply(400, {"error": {"code": 400, "message": "mock error"}})
        w, h = 512, 512
        if "x" in body.get("size", ""):
            w, h = (int(v) for v in body["size"].split("x"))
        alpha = 0 if body.get("background") == "transparent" else 255
        fmt = body.get("output_format", "png")
        buf = io.BytesIO()
        img = Image.new("RGBA", (w, h), (30, 144, 255, alpha))
        (img.convert("RGB") if fmt == "jpeg" else img).save(buf, format=fmt.upper())
        self.reply(200, {"created": 0, "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode(),
                                                  "media_type": f"image/{fmt}"}]})

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
