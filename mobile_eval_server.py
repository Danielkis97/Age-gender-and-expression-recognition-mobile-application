from __future__ import annotations

import argparse
import csv
import json
import socket
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "results" / "Results mobile metrics" / "mobile_browser_metrics.csv"
DEFAULT_MODEL = REPO_ROOT / "models" / "model.tflite"
DEFAULT_IMAGES_DIR = REPO_ROOT / "dataset" / "images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_HEADERS = [
    "timestamp_iso",
    "device_label",
    "user_agent",
    "run_id",
    "sample_idx",
    "image_name",
    "latency_ms",
    "model_url",
    "input_source",
    "notes",
]
CSV_HEADERS_LEGACY = [
    "timestamp_iso",
    "device_label",
    "user_agent",
    "run_id",
    "sample_idx",
    "latency_ms",
    "model_url",
    "input_source",
    "notes",
]


class MobileEvalHandler(SimpleHTTPRequestHandler):
    out_csv: Path = DEFAULT_CSV
    model_path: Path = DEFAULT_MODEL
    images_dir: Path = DEFAULT_IMAGES_DIR
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def end_headers(self) -> None:
        # Avoid stale mobile page/script caching during rapid local iterations.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            model_info = _read_tflite_input_info(self.model_path)
            self._send_json(
                {
                    "ok": True,
                    "server_time": datetime.now(timezone.utc).isoformat(),
                    "csv_path": str(self.out_csv),
                    "model_path": str(self.model_path),
                    "model_info": model_info,
                    "images_dir": str(self.images_dir),
                }
            )
            return
        if parsed.path == "/api/images":
            query = parse_qs(parsed.query or "")
            try:
                limit = int((query.get("limit") or ["20"])[0])
            except ValueError:
                limit = 20
            limit = max(1, min(limit, 500))

            if not self.images_dir.is_dir():
                self._send_json(
                    {
                        "ok": False,
                        "error": f"Images directory not found: {self.images_dir}",
                        "images": [],
                        "count": 0,
                    },
                    status=404,
                )
                return

            files = [
                p
                for p in sorted(self.images_dir.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ][:limit]
            images = [{"name": p.name, "url": f"/dataset/images/{quote(p.name)}"} for p in files]
            self._send_json({"ok": True, "count": len(images), "images": images})
            return
        if parsed.path in ("/mobile", "/mobile/"):
            self.path = "/mobile_browser_test/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/metrics":
            self._send_json({"ok": False, "error": "Unknown endpoint"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json({"ok": False, "error": "Empty body"}, status=400)
            return

        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "Invalid JSON"}, status=400)
            return

        if isinstance(payload, dict):
            records = [payload]
        elif isinstance(payload, list):
            records = payload
        else:
            self._send_json({"ok": False, "error": "Body must be object or list"}, status=400)
            return

        cleaned_rows: list[dict] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for rec in records:
            if not isinstance(rec, dict):
                continue
            try:
                latency_ms = float(rec.get("latency_ms"))
            except (TypeError, ValueError):
                continue

            row = {
                "timestamp_iso": str(rec.get("timestamp_iso") or now_iso),
                "device_label": str(rec.get("device_label") or "mobile-browser"),
                "user_agent": str(rec.get("user_agent") or ""),
                "run_id": str(rec.get("run_id") or ""),
                "sample_idx": str(rec.get("sample_idx") or ""),
                "image_name": str(rec.get("image_name") or ""),
                "latency_ms": f"{latency_ms:.4f}",
                "model_url": str(rec.get("model_url") or ""),
                "input_source": str(rec.get("input_source") or ""),
                "notes": str(rec.get("notes") or ""),
            }
            cleaned_rows.append(row)

        if not cleaned_rows:
            self._send_json({"ok": False, "error": "No valid metric rows"}, status=400)
            return

        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = _decide_csv_fieldnames(self.out_csv)
        with self.lock:
            write_header = not self.out_csv.exists()
            with self.out_csv.open("a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
                if write_header:
                    writer.writeheader()
                rows_to_write: list[dict] = []
                for row in cleaned_rows:
                    out_row = dict(row)
                    if "image_name" not in fieldnames:
                        image_name = out_row.pop("image_name", "")
                        if image_name:
                            note = out_row.get("notes") or ""
                            prefix = f"image_name={image_name}"
                            out_row["notes"] = f"{prefix}; {note}" if note else prefix
                    rows_to_write.append({k: out_row.get(k, "") for k in fieldnames})
                writer.writerows(rows_to_write)

        self._send_json({"ok": True, "rows_written": len(cleaned_rows), "csv_path": str(self.out_csv)})


def _local_ip_guess() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _decide_csv_fieldnames(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return CSV_HEADERS
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            first = f.readline().strip()
    except OSError:
        return CSV_HEADERS
    if not first:
        return CSV_HEADERS
    existing = [x.strip() for x in first.split(";") if x.strip()]
    if existing == CSV_HEADERS_LEGACY:
        return CSV_HEADERS_LEGACY
    if existing == CSV_HEADERS:
        return CSV_HEADERS
    if all(x in existing for x in CSV_HEADERS):
        return existing
    return CSV_HEADERS


def _read_tflite_input_info(model_path: Path) -> dict:
    if not model_path.is_file():
        return {"available": False, "error": f"Model not found: {model_path}"}
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": f"TensorFlow import failed: {exc}"}

    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        details = interpreter.get_input_details()[0]
        shape = [int(x) for x in details["shape"]]
        dtype_obj = details["dtype"]
        dtype = getattr(dtype_obj, "__name__", str(dtype_obj))
        return {"available": True, "shape": shape, "dtype": dtype}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": f"Interpreter failed: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve mobile browser benchmark page and collect latency metrics.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images_dir", type=Path, default=DEFAULT_IMAGES_DIR)
    args = parser.parse_args()

    MobileEvalHandler.out_csv = args.out_csv.resolve()
    MobileEvalHandler.model_path = args.model_path.resolve()
    MobileEvalHandler.images_dir = args.images_dir.resolve()
    server = ThreadingHTTPServer((args.host, args.port), MobileEvalHandler)

    ip = _local_ip_guess()
    print(f"[mobile_eval_server] Serving repo root: {REPO_ROOT}")
    print(f"[mobile_eval_server] Metrics CSV: {MobileEvalHandler.out_csv}")
    print(f"[mobile_eval_server] TFLite model: {MobileEvalHandler.model_path}")
    print(f"[mobile_eval_server] Images dir:   {MobileEvalHandler.images_dir}")
    print(f"[mobile_eval_server] Open on PC:     http://127.0.0.1:{args.port}/mobile")
    print(f"[mobile_eval_server] Open on iPhone: http://{ip}:{args.port}/mobile")
    print("[mobile_eval_server] Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
