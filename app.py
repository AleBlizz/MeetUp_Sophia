"""
Pure Python 3 web interface for KO/OK image similarity checking.
No external web framework needed — uses stdlib http.server only.

Run:  python3 app.py
Then open http://localhost:5000
"""

import os
import sys
import json
import email
import tempfile
import mimetypes
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from ko_ok_detector_es_inference import (
    get_embedding,
    knn_filtered,
    KO_THRESHOLD,
    KO_MARGIN,
    KO_MIN_VOTES,
    KNN_K,
    IMAGE_EXTENSIONS,
)

PORT        = int(os.getenv("WEB_PORT", 5000))
TEMPLATE_FR = Path(__file__).parent / "templates" / "index.html"
TEMPLATE_EN = Path(__file__).parent / "templates" / "index_en.html"
STATIC_DIR  = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Core check logic
# ---------------------------------------------------------------------------

def check_image(image_path: Path, filename: str) -> dict:
    embedding = get_embedding(image_path)
    ko_hits   = knn_filtered(embedding, "KO", include_image=True)
    ok_hits   = knn_filtered(embedding, "OK", include_image=True)

    top_ko = ko_hits[0]["_score"] if ko_hits else 0.0
    top_ok = ok_hits[0]["_score"] if ok_hits else 0.0
    avg_ko = sum(h["_score"] for h in ko_hits) / len(ko_hits) if ko_hits else 0.0
    avg_ok = sum(h["_score"] for h in ok_hits) / len(ok_hits) if ok_hits else 0.0
    votes  = sum(1 for h in ko_hits if h["_score"] >= KO_THRESHOLD)
    margin = avg_ko - avg_ok

    is_ko   = avg_ko >= KO_THRESHOLD and votes >= KO_MIN_VOTES and margin >= KO_MARGIN
    verdict = "KO" if is_ko else "OK"

    top_ko_label = ko_hits[0]["_source"]["defect_label"] if ko_hits else None
    defects = [top_ko_label] if (is_ko and top_ko_label) else []

    return {
        "filename": filename,
        "verdict":  verdict,
        "defects":  defects,
        "scores": {
            "top_ko":   round(top_ko, 4),
            "avg_ko":   round(avg_ko, 4),
            "ko_votes": votes,
            "top_ok":   round(top_ok, 4),
            "avg_ok":   round(avg_ok, 4),
            "margin":   round(margin, 4),
        },
        "thresholds": {
            "ko_threshold": KO_THRESHOLD,
            "ko_margin":    KO_MARGIN,
            "ko_min_votes": KO_MIN_VOTES,
            "knn_k":        KNN_K,
        },
        "ko_neighbours": [
            {
                "filename":        h["_source"]["filename"],
                "defect_label":    h["_source"]["defect_label"] or "KO",
                "score":           round(h["_score"], 4),
                "above_threshold": h["_score"] >= KO_THRESHOLD,
                "image_b64":       h["_source"].get("image_b64"),
            }
            for h in ko_hits
        ],
        "ok_neighbours": [
            {
                "filename":  h["_source"]["filename"],
                "score":     round(h["_score"], 4),
                "image_b64": h["_source"].get("image_b64"),
            }
            for h in ok_hits
        ],
    }


# ---------------------------------------------------------------------------
# Multipart parser (stdlib email module)
# ---------------------------------------------------------------------------

def parse_multipart(headers, body: bytes):
    """Return (filename, file_bytes) from a multipart/form-data body."""
    content_type = headers.get("Content-Type", "")
    raw = f"Content-Type: {content_type}\r\n\r\n".encode() + body
    msg = email.message_from_bytes(raw)
    for part in msg.walk():
        cd = part.get("Content-Disposition", "")
        if 'name="image"' in cd or "name=image" in cd:
            filename = part.get_filename() or "upload.jpg"
            return filename, part.get_payload(decode=True)
    return None, None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(fmt % args)

    def _send(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict):
        self._send(code, "application/json", json.dumps(data).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                self._send(200, "text/html; charset=utf-8", TEMPLATE_FR.read_bytes())
            except FileNotFoundError:
                self._send(404, "text/plain", b"templates/index.html not found")
        elif self.path in ("/en", "/en/"):
            try:
                self._send(200, "text/html; charset=utf-8", TEMPLATE_EN.read_bytes())
            except FileNotFoundError:
                self._send(404, "text/plain", b"templates/index_en.html not found")
        elif self.path.startswith("/static/"):
            file_path = STATIC_DIR / self.path[len("/static/"):]
            try:
                mime, _ = mimetypes.guess_type(str(file_path))
                self._send(200, mime or "application/octet-stream", file_path.read_bytes())
            except FileNotFoundError:
                self._send(404, "text/plain", b"static file not found")
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/api/check":
            self._send(404, "text/plain", b"not found")
            return

        length     = int(self.headers.get("Content-Length", 0))
        body       = self.rfile.read(length)
        filename, file_bytes = parse_multipart(self.headers, body)

        if not filename or not file_bytes:
            self._json(400, {"error": "No 'image' field found in request"})
            return

        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            self._json(400, {"error": f"Unsupported type '{suffix}'. Allowed: {', '.join(sorted(IMAGE_EXTENSIONS))}"})
            return

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(file_bytes)
            self._json(200, check_image(tmp_path, filename))
        except Exception as e:
            self._json(500, {"error": str(e)})
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server = HTTPServer(("", PORT), Handler)
    print(f"Listening on http://localhost:{PORT}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
