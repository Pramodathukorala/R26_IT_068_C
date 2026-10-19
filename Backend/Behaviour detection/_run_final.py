"""
============================================================================
 CHICKEN CCTV AI DETECTOR  —  v2 (streaming API + working alert endpoints)
============================================================================

 What runs
 ---------
   1. Gradio UI            http://127.0.0.1:786x
   2. Embedded HTTP API    http://<your-lan-ip>:8000
        GET  /                       -> link index (open this first)
        GET  /health                 -> {"ok": true, ...}
        GET  /api/status             -> latest AI result as JSON
        GET  /api/events?limit=50    -> recent detection events
        GET  /api/abnormal?limit=50  -> saved abnormal frames, newest first
        GET  /api/images             -> saved abnormal frames, listed
        GET  /api/images/<file>      -> one saved abnormal frame (JPEG)
        GET  /api/snapshot.jpg       -> single latest frame
        GET  /api/stream.mjpg        -> LIVE MJPEG camera stream
        GET  /view                   -> browser page showing the live stream
        POST /api/alerts/abnormal    -> built-in alert receiver
        POST /api/alerts/normal      -> built-in alert receiver
        GET  /api/received?limit=50  -> alerts this machine has received

   The two POST endpoints are the default alert targets, so alerting works
   out of the box with no external server. Point the UI fields at your own
   URL whenever you have one.

 Requirements
 ------------
   pip install -U ultralytics gradio opencv-python pandas torch requests
============================================================================
"""

from __future__ import annotations

import base64
import json
import os
import platform
import queue
import re
import socket
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, unquote

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import gradio as gr

# Shared captured-image store lives in Backend/ (one level above this notebook)
import sys
for _dir in (os.getcwd(), os.path.dirname(os.getcwd())):
    if os.path.exists(os.path.join(_dir, "image_store.py")) and _dir not in sys.path:
        sys.path.insert(0, _dir)
from image_store import save_image

try:
    from ultralytics import YOLO, YOLOWorld
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Ultralytics is not available.\n"
        "Install:  pip install -U ultralytics gradio opencv-python pandas torch requests"
    ) from exc


# ============================================================================
# 1. CONFIGURATION
# ============================================================================


def env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


# ---------------------------------------------------------------- camera ---
CAMERA_URL = env(
    "CHICKEN_CAMERA_URL",
    "rtsp://admin:Aa20020607@192.168.8.103:554/cam/realmonitor?channel=1&subtype=0",
)

# ------------------------------------------------------------- scheduling ---
CHECK_SECONDS = float(env("CHICKEN_CHECK_SECONDS", "30"))
ALERT_COOLDOWN_SECONDS = float(env("CHICKEN_ALERT_COOLDOWN", "30"))
COOLDOWN_BYPASS_ON_STATE_CHANGE = True
UI_FPS = 6

# ------------------------------------------------------------- API server ---
API_SERVER_HOST = env("CHICKEN_API_HOST", "0.0.0.0")   # 0.0.0.0 = reachable from LAN
API_SERVER_PORT = int(env("CHICKEN_API_PORT", "8000"))
STREAM_FPS = float(env("CHICKEN_STREAM_FPS", "10"))
STREAM_QUALITY = int(env("CHICKEN_STREAM_QUALITY", "70"))
STREAM_MAX_WIDTH = int(env("CHICKEN_STREAM_WIDTH", "960"))
# Optional shared secret. If set, every /api/* call needs ?token=... or
# an Authorization: Bearer header.
API_ACCESS_TOKEN = env("CHICKEN_API_ACCESS_TOKEN", "")

# ----------------------------------------------------------------- models ---
PRESENCE_MODEL_PATH = env("CHICKEN_PRESENCE_MODEL", "yolov8s-worldv2.pt")
CLASSIFIER_MODEL_PATH = env("CHICKEN_CLASSIFIER_MODEL", r"D:\My backend\chicken_best.pt")

PRESENCE_CLASSES = ["chicken", "hen", "rooster", "domestic chicken", "poultry"]
PRESENCE_CONFIDENCE = 0.15
CLASSIFIER_MIN_CONFIDENCE = 0.30
MIN_CHICKEN_BOX_AREA_RATIO = 0.0005

MODEL_IMGSZ = 640
PRESENCE_RECENT_FRAMES = 6
FRAME_SAMPLE_SECONDS = 1.0

# ----------------------------------------------------------------- output ---
OUTPUT_FOLDER = env("CHICKEN_OUTPUT_DIR", r"D:\My backend\chicken\Chicken_CCTV_Abnormal_Events")
ABNORMAL_SUBDIR = "abnormal_saved"
API_LOG_FILE = "api_log.jsonl"
API_SUMMARY_FILE = "api_summary.json"
RECEIVED_LOG_FILE = "received_alerts.jsonl"

# -------------------------------------------------------------------- API ---
# Default to this machine's own receiver so the links work immediately.
DEFAULT_ABNORMAL_PATH = "/api/alerts/abnormal"
DEFAULT_NORMAL_PATH = "/api/alerts/normal"
ABNORMAL_API_URL = env("CHICKEN_ABNORMAL_API_URL", "")   # blank -> auto local URL
NORMAL_API_URL = env("CHICKEN_NORMAL_API_URL", "")       # blank -> auto local URL
API_AUTH_TOKEN = env("CHICKEN_API_TOKEN", "")

# ------------------------------------------------------------ FARM EVENTS ---
# FARM EVENTS - cross-component integration.
# The disease and predator detectors run as their own services on their own
# ports. We push every check to them and merge what they push back into
# /api/feed. Contract: Backend/FARM_EVENTS.md.
FARM_SERVICE_NAME = "chicken"
FARM_EVENTS_ENABLED = env("FARM_EVENTS", "1") != "0"
FARM_PEER_INBOXES = {
    "disease": env("FARM_PEER_DISEASE", "http://127.0.0.1:8001/api/events/external"),
    "predator": env("FARM_PEER_PREDATOR", "http://127.0.0.1:8002/api/events/external"),
}
FARM_PEER_TOKEN = env("FARM_PEER_TOKEN", "")
FARM_PEER_TIMEOUT = float(env("FARM_PEER_TIMEOUT", "3.0"))

API_TIMEOUT_SECONDS = 10
API_MAX_RETRIES = 3
API_RETRY_BACKOFF = 2.0
SEND_IMAGE_BASE64 = True
IMAGE_JPEG_QUALITY = 70
IMAGE_MAX_WIDTH = 640

PLACEHOLDER_HOST_TOKENS = (
    "your-actual", "your-api", "your-endpoint", "example.com",
    "changeme", "placeholder", "todo", "xxx",
)

# --------------------------------------------------------------- display ---
MAX_WIDTH, MAX_HEIGHT = 960, 540

# ---------------------------------------------------------------- device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"

try:
    GRADIO_MAJOR = int(str(gr.__version__).split(".")[0])
except Exception:
    GRADIO_MAJOR = 6


# ============================================================================
# 2. HELPERS
# ============================================================================


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def format_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def normalize_name(name: Any) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def safe_model_names(names: Any) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(list(names))}


def resize_frame(frame: np.ndarray, max_w: int = MAX_WIDTH, max_h: int = MAX_HEIGHT) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def mask_credentials(url: str) -> str:
    try:
        parts = urlparse(str(url))
        if parts.password:
            netloc = f"{parts.username}:***@{parts.hostname}"
            if parts.port:
                netloc += f":{parts.port}"
            return urlunparse(parts._replace(netloc=netloc))
    except Exception:
        pass
    return str(url)


def gr_skip():
    return gr.skip() if hasattr(gr, "skip") else gr.update()


def get_lan_ip() -> str:
    """Best-effort local network IP (the one other devices should use)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))       # no traffic is sent
        return sock.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    finally:
        sock.close()


LAN_IP = get_lan_ip()


def api_base_url(host: str = "127.0.0.1") -> str:
    return f"http://{host}:{API_SERVER_PORT}"


def encode_jpeg(frame: np.ndarray, quality: int, max_width: int) -> Optional[bytes]:
    try:
        small = resize_frame(frame, max_width, max_width)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None
    except Exception:
        return None


def frame_to_base64_jpeg(frame: np.ndarray) -> Optional[str]:
    raw = encode_jpeg(frame, IMAGE_JPEG_QUALITY, IMAGE_MAX_WIDTH)
    return base64.b64encode(raw).decode("ascii") if raw else None


# ============================================================================
# 3. SHARED RUNTIME STATE
#    The HTTP server and the detection loop both read/write this.
# ============================================================================


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._raw: Optional[np.ndarray] = None        # clean camera frame
        self._annotated: Optional[np.ndarray] = None  # last AI-annotated frame
        self._frame_seq = 0
        self.started_at = time.monotonic()
        self.status: Dict[str, Any] = {
            "running": False, "camera_connected": False, "result": "WAITING",
            "confidence": 0.0, "chicken_count": 0, "check_number": 0,
            "next_check_in": None, "last_check_at": None, "last_alert": None,
            "presence_details": None, "classifier_details": None, "reconnects": 0,
        }
        self.events: deque = deque(maxlen=500)
        self.received: deque = deque(maxlen=500)
        self.output_folder = OUTPUT_FOLDER

    # ------------------------------------------------------------- frames --
    def set_raw(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        with self._lock:
            self._raw = frame
            self._frame_seq += 1

    def set_annotated(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        with self._lock:
            self._annotated = frame.copy()

    def get_frame(self, annotated: bool = False) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            frame = self._annotated if (annotated and self._annotated is not None) else self._raw
            return (None if frame is None else frame), self._frame_seq

    # ------------------------------------------------------------- status --
    def update_status(self, **kwargs) -> None:
        with self._lock:
            self.status.update(kwargs)

    def snapshot_status(self) -> Dict[str, Any]:
        with self._lock:
            data = dict(self.status)
        data["uptime_seconds"] = round(time.monotonic() - self.started_at, 1)
        data["server_time"] = now_iso()
        return data

    def add_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.events.appendleft(event)

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.events)[:max(1, min(limit, 500))]

    # ------------------------------------------------- received alert log --
    def add_received(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            self.received.appendleft(entry)
            folder = self.output_folder
        try:
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, RECEIVED_LOG_FILE), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    def recent_received(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.received)[:max(1, min(limit, 500))]


RUNTIME = RuntimeState()


# ============================================================================
# 3b. SAVED ABNORMAL FRAMES  (read-only)
# ============================================================================
# Every ABNORMAL check already writes an annotated JPEG into
# <OUTPUT_FOLDER>/abnormal_saved. Nothing below changes that, or the detection
# that produces it - this block only lets a client read those files back over
# HTTP, so the mobile app never has to see a local disk path.

ABNORMAL_NAME_RE = re.compile(r"^abnormal_(\d{1,6})_(\d{8})_(\d{6})\.jpg$")


def abnormal_dir() -> str:
    return os.path.join(RUNTIME.output_folder, ABNORMAL_SUBDIR)


def parse_abnormal_name(name: str) -> Optional[Dict[str, Any]]:
    """abnormal_0115_20260901_001737.jpg -> its check number and capture time."""
    match = ABNORMAL_NAME_RE.match(name or "")
    if not match:
        return None
    check, day, clock = match.groups()
    try:
        when = datetime.strptime(day + clock, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return {"check_number": int(check),
            "captured_at": when.astimezone().isoformat(timespec="seconds")}


def image_url(name: str) -> str:
    return f"{api_base_url(LAN_IP)}/api/images/{name}"


def read_abnormal_image(name: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Bytes of one saved frame, or (None, reason). The name is reduced to a bare
    filename and must match ABNORMAL_NAME_RE, so a caller cannot walk out of
    the folder with ../ or an absolute path.
    """
    safe = os.path.basename(str(name or ""))
    if not ABNORMAL_NAME_RE.match(safe):
        return None, "invalid image name"
    folder = os.path.abspath(abnormal_dir())
    full = os.path.abspath(os.path.join(folder, safe))
    if os.path.dirname(full) != folder:
        return None, "outside the image folder"
    if not os.path.isfile(full):
        return None, "image not found"
    try:
        with open(full, "rb") as handle:
            return handle.read(), None
    except OSError as exc:
        return None, f"read failed: {exc}"


def list_abnormal_images(limit: int = 100) -> List[Dict[str, Any]]:
    """Saved frames, newest first, described by URL rather than by path."""
    folder = abnormal_dir()
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    items: List[Dict[str, Any]] = []
    for name in names:
        info = parse_abnormal_name(name)
        if info is None:
            continue
        try:
            stat = os.stat(os.path.join(folder, name))
        except OSError:
            continue
        items.append({
            "filename": name,
            "image_url": image_url(name),
            "check_number": info["check_number"],
            "captured_at": info["captured_at"],
            "size_bytes": stat.st_size,
            "saved_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        })
    items.sort(key=lambda item: item["saved_at"], reverse=True)
    return items[:max(1, min(int(limit or 100), 500))]


def public_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """An /api/events row with the on-disk path swapped for a URL."""
    out = {key: value for key, value in event.items() if key != "image_path"}
    name = os.path.basename(str(event.get("image_path") or "")) or None
    out["image_filename"] = name
    out["image_url"] = image_url(name) if name else None
    return out


def abnormal_history(limit: int = 50) -> Dict[str, Any]:
    """
    Newest-first history for the mobile alert gallery: one row per saved frame,
    carrying the check that produced it whenever that check is still in the
    in-memory log (that log holds 500 checks; the files outlive it, and outlive
    a restart, so the folder is what the list is built from).
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    for event in RUNTIME.recent_events(500):
        name = os.path.basename(str(event.get("image_path") or ""))
        if name:
            by_name.setdefault(name, event)

    rows: List[Dict[str, Any]] = []
    for image in list_abnormal_images(500):
        event = by_name.get(image["filename"], {})
        rows.append({**image,
                     "result": event.get("result", "ABNORMAL"),
                     "confidence": event.get("confidence"),
                     "chicken_count": event.get("chicken_count"),
                     "alert": event.get("alert"),
                     "timestamp": event.get("timestamp") or image["captured_at"]})
    return {"count": len(rows),
            "events": rows[:max(1, min(int(limit or 50), 500))]}


# ============================================================================
# 4. EMBEDDED HTTP API  (streaming + alert receiver)
# ============================================================================

VIEW_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Chicken CCTV — live</title>
<style>
 body{{margin:0;background:#0b0b0d;color:#e6e6e6;font:15px/1.5 system-ui,sans-serif}}
 header{{padding:14px 20px;border-bottom:1px solid #24242a}}
 h1{{margin:0;font-size:17px;font-weight:600}}
 main{{padding:20px;display:flex;gap:20px;flex-wrap:wrap}}
 img{{max-width:100%;border-radius:12px;background:#000;border:1px solid #24242a}}
 aside{{min-width:260px;flex:1}}
 .k{{color:#8a8a94}} .v{{font-weight:600}}
 .row{{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1c1c22}}
 .ABNORMAL{{color:#ff5c5c}} .NORMAL{{color:#4ade80}} .UNDETECTED{{color:#fbbf24}}
</style></head><body>
<header><h1>🐔 Chicken CCTV — live stream</h1></header>
<main>
  <img src="/api/stream.mjpg{q}" alt="live stream">
  <aside id="s"></aside>
</main>
<script>
async function tick(){{
  try{{
    const r = await fetch('/api/status{q}');
    const d = await r.json();
    document.getElementById('s').innerHTML = Object.entries({{
      'Result': `<span class="${{d.result}}">${{d.result}}</span>`,
      'Confidence': (d.confidence*100).toFixed(1)+'%',
      'Chickens': d.chicken_count,
      'Check #': d.check_number,
      'Next check in': d.next_check_in ?? '-',
      'Camera': d.camera_connected ? 'connected' : 'offline',
      'Last alert': d.last_alert ?? '-'
    }}).map(([k,v])=>`<div class="row"><span class="k">${{k}}</span><span class="v">${{v}}</span></div>`).join('');
  }}catch(e){{}}
}}
tick(); setInterval(tick, 2000);
</script></body></html>"""


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "ChickenCCTV/2.0"
    protocol_version = "HTTP/1.1"

    # ----------------------------------------------------------- plumbing --
    def log_message(self, fmt: str, *args) -> None:      # silence per-request spam
        return

    def _authorized(self, params: Dict[str, List[str]]) -> bool:
        if not API_ACCESS_TOKEN:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and header[7:].strip() == API_ACCESS_TOKEN:
            return True
        return params.get("token", [""])[0] == API_ACCESS_TOKEN

    def _send(self, code: int, body: bytes, content_type: str,
              extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload, default=str, indent=2).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:                        # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    # ---------------------------------------------------------------- GET --
    def do_GET(self) -> None:                            # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        if route in ("/", "/health"):
            if route == "/health":
                return self._json(200, {"ok": True, "service": "chicken-cctv",
                                        "time": now_iso(),
                                        "camera_connected": RUNTIME.status["camera_connected"]})
            return self._send(200, self._index_page().encode("utf-8"), "text/html; charset=utf-8")

        if not self._authorized(params):
            return self._json(401, {"error": "unauthorized — supply ?token= or a Bearer header"})

        if route == "/view":
            qs = f"?token={API_ACCESS_TOKEN}" if API_ACCESS_TOKEN else ""
            return self._send(200, VIEW_PAGE.format(q=qs).encode("utf-8"), "text/html; charset=utf-8")

        if route == "/api/status":
            return self._json(200, RUNTIME.snapshot_status())

        if route == "/api/events":
            limit = int(params.get("limit", ["50"])[0] or 50)
            events = [public_event(item) for item in RUNTIME.recent_events(limit)]
            return self._json(200, {"count": len(RUNTIME.events), "events": events})

        if route == "/api/abnormal":
            limit = int(params.get("limit", ["50"])[0] or 50)
            return self._json(200, abnormal_history(limit))

        if route == "/api/images":
            limit = int(params.get("limit", ["100"])[0] or 100)
            images = list_abnormal_images(limit)
            return self._json(200, {"count": len(images), "images": images})

        if route.startswith("/api/images/"):
            name = unquote(route[len("/api/images/"):])
            data, problem = read_abnormal_image(name)
            if data is None:
                return self._json(404, {"error": problem, "filename": name})
            return self._send(200, data, "image/jpeg",
                              {"Cache-Control": "public, max-age=86400"})

        if route == "/api/received":
            limit = int(params.get("limit", ["50"])[0] or 50)
            return self._json(200, {"count": len(RUNTIME.received),
                                    "alerts": RUNTIME.recent_received(limit)})

        if route == "/api/feed":
            limit = int(params.get("limit", ["50"])[0] or 50)
            return self._json(200, farm_feed(limit))

        if route == "/api/feed/peers":
            return self._json(200, {"service": FARM_SERVICE_NAME,
                                    "peers": FARM_PEERS.status()})

        if route in ("/api/snapshot.jpg", "/api/snapshot"):
            annotated = params.get("annotated", ["0"])[0] in ("1", "true", "yes")
            frame, _ = RUNTIME.get_frame(annotated)
            if frame is None:
                return self._json(503, {"error": "no frame yet — start detection first"})
            raw = encode_jpeg(frame, STREAM_QUALITY, STREAM_MAX_WIDTH)
            if raw is None:
                return self._json(500, {"error": "jpeg encode failed"})
            return self._send(200, raw, "image/jpeg")

        if route in ("/api/stream.mjpg", "/api/stream"):
            annotated = params.get("annotated", ["0"])[0] in ("1", "true", "yes")
            return self._stream_mjpeg(annotated)

        return self._json(404, {"error": "unknown route", "see": "/"})

    # --------------------------------------------------------------- POST --
    def do_POST(self) -> None:                           # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        if not self._authorized(params):
            return self._json(401, {"error": "unauthorized"})

        if route not in ("/api/alerts/abnormal", "/api/alerts/normal", "/api/alerts"):
            return self._json(404, {"error": "unknown route", "see": "/"})

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            return self._json(400, {"error": f"invalid JSON: {exc}"})

        kind = route.rsplit("/", 1)[-1]
        if kind == "alerts":
            kind = str(payload.get("result", "unknown")).lower()

        stored = dict(payload)
        image_b64 = stored.pop("image_base64", None)
        entry = {
            "received_at": now_iso(),
            "kind": kind,
            "from": self.client_address[0],
            "has_image": bool(image_b64),
            "image_bytes": len(image_b64) if image_b64 else 0,
            "payload": stored,
        }
        RUNTIME.add_received(entry)
        print(f"📥 alert received on /api/alerts/{kind}  "
              f"(check #{payload.get('check_number')}, {entry['image_bytes']} b64 bytes)")

        return self._json(200, {"ok": True, "kind": kind, "received_at": entry["received_at"]})

    # ------------------------------------------------------------- MJPEG ---
    def _stream_mjpeg(self, annotated: bool) -> None:
        boundary = "chickenframe"
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        interval = 1.0 / max(1.0, STREAM_FPS)
        last_seq = -1
        idle_since = time.monotonic()

        try:
            while True:
                frame, seq = RUNTIME.get_frame(annotated)
                if frame is None:
                    if time.monotonic() - idle_since > 30:
                        break                       # nothing to stream for 30s
                    time.sleep(0.2)
                    continue

                idle_since = time.monotonic()
                if seq == last_seq:                 # don't resend an identical frame
                    time.sleep(interval / 2)
                    continue
                last_seq = seq

                jpeg = encode_jpeg(frame, STREAM_QUALITY, STREAM_MAX_WIDTH)
                if jpeg is None:
                    time.sleep(interval)
                    continue

                self.wfile.write(f"--{boundary}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                     # viewer closed the tab
        except Exception as exc:
            print(f"⚠️ stream ended: {exc}")

    # -------------------------------------------------------- index page ---
    def _index_page(self) -> str:
        qs = f"?token={API_ACCESS_TOKEN}" if API_ACCESS_TOKEN else ""
        base_lan = api_base_url(LAN_IP)
        rows = [
            ("Live stream page", f"/view{qs}", "open in a browser"),
            ("MJPEG stream", f"/api/stream.mjpg{qs}", "use in VLC, an <img> tag, or another app"),
            ("MJPEG + AI boxes", f"/api/stream.mjpg?annotated=1", "annotated frames"),
            ("Snapshot JPEG", f"/api/snapshot.jpg{qs}", "single frame"),
            ("Status JSON", f"/api/status{qs}", "current AI result"),
            ("Events JSON", f"/api/events?limit=50", "recent checks"),
            ("Received alerts", f"/api/received?limit=50", "what this receiver got"),
            ("Health", "/health", "liveness probe"),
        ]
        items = "".join(
            f'<li><a href="{path}">{name}</a> <code>{base_lan}{path}</code>'
            f'<span> — {note}</span></li>' for name, path, note in rows)
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Chicken CCTV API</title><style>
body{{font:15px/1.7 system-ui,sans-serif;margin:40px;max-width:820px;color:#1a1a1a}}
li{{margin:.5em 0}} code{{background:#f2f2f4;padding:2px 6px;border-radius:5px;font-size:13px}}
span{{color:#666}} h1{{font-size:20px}}</style></head><body>
<h1>🐔 Chicken CCTV API</h1>
<p>LAN base URL: <code>{base_lan}</code></p>
<ul>{items}</ul>
<h3>Alert receivers (POST)</h3>
<ul><li><code>{base_lan}/api/alerts/abnormal</code></li>
<li><code>{base_lan}/api/alerts/normal</code></li></ul>
</body></html>"""


class ApiServer:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self) -> bool:
        for attempt in range(10):                    # walk the port if it's taken
            port = self.port + attempt
            try:
                server = ThreadingHTTPServer((self.host, port), ApiHandler)
                server.daemon_threads = True
                self._server = server
                self.port = port
                globals()["API_SERVER_PORT"] = port
                self._thread = threading.Thread(target=server.serve_forever,
                                                name="api-server", daemon=True)
                self._thread.start()
                return True
            except OSError as exc:
                self.error = str(exc)
                continue
        print(f"❌ could not start API server: {self.error}")
        return False

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()


API_SERVER = ApiServer(API_SERVER_HOST, API_SERVER_PORT)


# ============================================================================
# 4b. FARM EVENTS  (talking to the disease and predator services)
# ============================================================================


class FarmPeers:
    """
    Fire-and-forget delivery of our events to the other two detectors.

    Three rules, shared with the other services:
      * received events are never re-published - that is what stops the three
        of them flooding each other;
      * delivery never blocks the detection loop - it runs on this worker;
      * a peer being down is normal, counted, and never fatal.
    """

    def __init__(self) -> None:
        self.own: deque = deque(maxlen=500)
        self.state = {name: {"url": url, "sent": 0, "failed": 0,
                             "last_ok": None, "last_error": None}
                      for name, url in FARM_PEER_INBOXES.items() if url}
        self.seq = 0
        self._lock = threading.Lock()
        self._queue: "queue.Queue" = queue.Queue(maxsize=200)
        self._session = requests.Session()
        threading.Thread(target=self._loop, name="farm-peers", daemon=True).start()

    def publish(self, kind: str, message: str, severity: str = "info",
                data: Optional[dict] = None) -> Optional[dict]:
        """Record one of our events and push it to the peers."""
        if not FARM_EVENTS_ENABLED:
            return None

        with self._lock:
            self.seq += 1
            seq = self.seq

        event = {
            "id": f"{FARM_SERVICE_NAME}-{time.strftime('%Y%m%d_%H%M%S')}-{seq:04d}",
            "source": FARM_SERVICE_NAME,
            "kind": kind,
            "severity": severity if severity in ("info", "warning", "critical") else "info",
            "message": message,
            "timestamp": now_iso(),
            "data": data or {},
        }

        with self._lock:
            self.own.appendleft(event)

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            print("⚠️  farm peer queue full — dropped fan-out for", event["id"])

        return event

    def status(self) -> dict:
        with self._lock:
            return {name: dict(state) for name, state in self.state.items()}

    def _loop(self) -> None:
        while True:
            event = self._queue.get()
            headers = {"Content-Type": "application/json"}
            if FARM_PEER_TOKEN:
                headers["Authorization"] = f"Bearer {FARM_PEER_TOKEN}"

            for state in self.state.values():
                try:
                    resp = self._session.post(state["url"], json=event,
                                              headers=headers, timeout=FARM_PEER_TIMEOUT)
                    if resp.status_code < 400:
                        state["sent"] += 1
                        state["last_ok"] = now_iso()
                        state["last_error"] = None
                    else:
                        state["failed"] += 1
                        state["last_error"] = f"HTTP {resp.status_code}"
                except Exception as exc:
                    # A peer that simply is not running is the normal case.
                    state["failed"] += 1
                    state["last_error"] = f"{type(exc).__name__}: {exc}"


FARM_PEERS = FarmPeers()


def farm_feed(limit: int = 50) -> dict:
    """Our checks and the peers' events, merged, newest first."""
    with FARM_PEERS._lock:
        own = list(FARM_PEERS.own)

    peers = []
    for entry in RUNTIME.recent_received(500):
        body = entry.get("payload") or {}
        peers.append({
            "id": str(body.get("id") or f"{body.get('source', 'peer')}-{entry.get('received_at')}"),
            "source": str(body.get("source") or entry.get("kind") or "peer"),
            "kind": str(body.get("kind") or body.get("event") or "peer_event"),
            "severity": body.get("severity") if body.get("severity") in
                        ("info", "warning", "critical") else "info",
            "message": str(body.get("message") or f"event from {body.get('source', 'a peer')}"),
            "timestamp": str(body.get("timestamp") or entry.get("received_at") or ""),
            "received_at": entry.get("received_at"),
            "data": body,
        })

    merged = own + peers
    merged.sort(key=lambda e: str(e.get("timestamp", "")), reverse=True)

    return {"service": FARM_SERVICE_NAME,
            "counts": {"own": len(own), "received": len(peers)},
            "events": merged[:max(1, min(int(limit), 1000))]}


# ============================================================================
# 5. ALERT LAYER
# ============================================================================


@dataclass
class Endpoint:
    name: str
    raw_url: str
    url: Optional[str] = None
    error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.url is not None

    @property
    def status_text(self) -> str:
        return f"✅ `{self.url}`" if self.ready else f"⛔ not usable — {self.error}"


def build_endpoint(name: str, raw_url: str, fallback: Optional[str] = None) -> Endpoint:
    """Validate a URL; fall back to the built-in local receiver when blank."""
    url = str(raw_url or "").strip()
    if not url and fallback:
        url = fallback

    if not url:
        return Endpoint(name, raw_url, None, "no URL configured")

    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        return Endpoint(name, raw_url, None, "URL must start with http:// or https://")
    if not parts.netloc:
        return Endpoint(name, raw_url, None, "URL has no host")
    for token in PLACEHOLDER_HOST_TOKENS:
        if token in parts.netloc.lower():
            return Endpoint(name, raw_url, None,
                            f"placeholder host ('{token}') — replace with a real endpoint")

    return Endpoint(name, raw_url, url, None)


class CooldownGate:
    """First alert always fires; a state change fires; otherwise mute for N seconds."""

    def __init__(self, seconds: float, bypass_on_change: bool = True):
        self.seconds = float(seconds)
        self.bypass_on_change = bypass_on_change
        self._last_sent: Dict[str, float] = {}
        self._last_kind: Optional[str] = None
        self._suppressed: Dict[str, int] = {}
        self._lock = threading.Lock()

    def check(self, kind: str) -> Tuple[bool, str]:
        with self._lock:
            now = time.monotonic()
            changed = kind != self._last_kind
            last = self._last_sent.get(kind)
            elapsed = None if last is None else now - last

            if last is None:
                allow, reason = True, "first alert of this type"
            elif changed and self.bypass_on_change:
                allow, reason = True, f"state changed {self._last_kind} → {kind}"
            elif elapsed >= self.seconds:
                allow, reason = True, f"cooldown elapsed ({elapsed:.0f}s)"
            else:
                allow, reason = False, f"cooldown active — {self.seconds - elapsed:.0f}s left"

            self._last_kind = kind
            if allow:
                self._last_sent[kind] = now
            else:
                self._suppressed[kind] = self._suppressed.get(kind, 0) + 1
            return allow, reason

    def remaining(self, kind: str) -> float:
        with self._lock:
            last = self._last_sent.get(kind)
            return 0.0 if last is None else max(0.0, self.seconds - (time.monotonic() - last))


class AlertDispatcher:
    """Background POSTs with retries + append-only JSONL log."""

    def __init__(self, log_dir: str, token: str = "", cooldown: float = ALERT_COOLDOWN_SECONDS):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, API_LOG_FILE)
        self.summary_path = os.path.join(log_dir, API_SUMMARY_FILE)
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

        self.gate = CooldownGate(cooldown, COOLDOWN_BYPASS_ON_STATE_CHANGE)
        self.summary = {"sent": 0, "failed": 0, "skipped_cooldown": 0,
                        "skipped_unconfigured": 0, "by_type": {}, "last_error": None}

        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._session = requests.Session()
        threading.Thread(target=self._worker, name="alert-dispatcher", daemon=True).start()

    def submit(self, endpoint: Endpoint, kind: str, payload: dict) -> Tuple[bool, str]:
        if not endpoint.ready:
            self._bump("skipped_unconfigured", kind)
            self._write_log({"timestamp": now_iso(), "type": kind, "outcome": "skipped",
                             "reason": endpoint.error})
            return False, f"skipped — {endpoint.error}"

        allow, reason = self.gate.check(kind)
        if not allow:
            self._bump("skipped_cooldown", kind)
            self._write_log({"timestamp": now_iso(), "type": kind, "outcome": "muted",
                             "reason": reason, "url": endpoint.url})
            return False, f"muted — {reason}"

        try:
            self._queue.put_nowait({"endpoint": endpoint, "kind": kind, "payload": payload})
        except queue.Full:
            self._bump("failed", kind)
            return False, "alert queue full"
        return True, f"queued — {reason}"

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.summary))

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                self._deliver(job)
            except Exception:
                traceback.print_exc()
            finally:
                self._queue.task_done()

    def _deliver(self, job: dict) -> None:
        endpoint: Endpoint = job["endpoint"]
        kind, payload = job["kind"], job["payload"]
        attempts: List[dict] = []

        for attempt in range(1, API_MAX_RETRIES + 1):
            started = time.perf_counter()
            try:
                resp = self._session.post(endpoint.url, json=payload,
                                          headers=self.headers, timeout=API_TIMEOUT_SECONDS)
                ms = round((time.perf_counter() - started) * 1000, 1)
                attempts.append({"attempt": attempt, "status": resp.status_code,
                                 "ms": ms, "body": resp.text[:400]})
                if resp.ok:
                    self._bump("sent", kind)
                    self._write_log({"timestamp": now_iso(), "type": kind, "outcome": "sent",
                                     "url": endpoint.url, "attempts": attempts,
                                     "payload": self._log_payload(payload)})
                    print(f"✅ {kind.upper()} alert delivered ({resp.status_code}, {ms} ms)")
                    return
                if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
                    break
            except requests.RequestException as exc:
                attempts.append({"attempt": attempt, "error": str(exc)[:300]})

            if attempt < API_MAX_RETRIES:
                time.sleep(API_RETRY_BACKOFF * attempt)

        self._bump("failed", kind, error=str(attempts[-1] if attempts else "unknown"))
        self._write_log({"timestamp": now_iso(), "type": kind, "outcome": "failed",
                         "url": endpoint.url, "attempts": attempts,
                         "payload": self._log_payload(payload)})
        print(f"❌ {kind.upper()} alert failed after {len(attempts)} attempt(s)")

    @staticmethod
    def _log_payload(payload: dict) -> dict:
        trimmed = dict(payload)
        if trimmed.get("image_base64"):
            trimmed["image_base64"] = f"<{len(trimmed['image_base64'])} chars omitted>"
        return trimmed

    def _bump(self, counter: str, kind: str, error: Optional[str] = None) -> None:
        with self._lock:
            self.summary[counter] = self.summary.get(counter, 0) + 1
            per_type = self.summary["by_type"].setdefault(
                kind, {"sent": 0, "failed": 0, "skipped_cooldown": 0, "skipped_unconfigured": 0})
            per_type[counter] = per_type.get(counter, 0) + 1
            if error:
                self.summary["last_error"] = error[:300]
            try:
                with open(self.summary_path, "w", encoding="utf-8") as fh:
                    json.dump(self.summary, fh, indent=2)
            except Exception:
                pass

    def _write_log(self, entry: dict) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            print(f"⚠️ could not append to API log: {exc}")


def build_payload(check_number: int, result: str, chicken_count: int,
                  presence_details: str, classifier_details: str, confidence: float,
                  camera_url: str, image_path: Optional[str] = None,
                  frame: Optional[np.ndarray] = None) -> dict:
    payload = {
        "event": f"chicken.{result.lower()}",
        "timestamp": now_iso(),
        "check_number": check_number,
        "result": result,
        "confidence": round(float(confidence), 4),
        "chicken_count": chicken_count,
        "presence_details": presence_details,
        "classifier_details": classifier_details,
        "camera": mask_credentials(camera_url),
        "host": socket.gethostname(),
        "stream_url": f"{api_base_url(LAN_IP)}/api/stream.mjpg",
        "snapshot_url": f"{api_base_url(LAN_IP)}/api/snapshot.jpg",
        "image_path": image_path,
    }
    if SEND_IMAGE_BASE64 and frame is not None and result == "ABNORMAL":
        encoded = frame_to_base64_jpeg(frame)
        if encoded:
            payload["image_base64"] = encoded
            payload["image_mime"] = "image/jpeg"
    return payload


# ============================================================================
# 6. CAMERA
# ============================================================================


class CameraStream:
    def __init__(self, url: str, sample_seconds: float = FRAME_SAMPLE_SECONDS,
                 history: int = PRESENCE_RECENT_FRAMES):
        self.url = str(url).strip()
        self.sample_seconds = sample_seconds
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[np.ndarray] = None
        self._history: deque = deque(maxlen=max(2, history))
        self._last_sample = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self.reconnects = 0
        self.connected = False
        self.publish = True          # mirror frames into RUNTIME for the HTTP API

    def open(self, timeout: float = 10.0) -> Tuple[bool, str]:
        cap = cv2.VideoCapture(self.url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return False, "camera could not be opened (check URL, credentials, network)"

        deadline = time.time() + timeout
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                self._cap = cap
                self.connected = True
                self.error = None
                self._store(frame)
                return True, f"connected — {frame.shape[1]}x{frame.shape[0]}"
            time.sleep(0.1)

        cap.release()
        return False, "camera opened but delivered no frames within the timeout"

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self.connected = False
        RUNTIME.update_status(camera_connected=False)

    def _store(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        with self._lock:
            self._latest = frame
            if now - self._last_sample >= self.sample_seconds:
                self._history.append(frame.copy())
                self._last_sample = now
        if self.publish:
            RUNTIME.set_raw(frame)

    def _loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            if self._cap is None:
                self._reconnect()
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None or not frame.size:
                failures += 1
                if failures >= 40:
                    failures = 0
                    self._reconnect()
                time.sleep(0.02)
                continue

            failures = 0
            self._store(frame)

    def _reconnect(self) -> None:
        self.connected = False
        self.reconnects += 1
        RUNTIME.update_status(camera_connected=False, reconnects=self.reconnects)
        print(f"⚠️ camera lost — reconnect attempt #{self.reconnects}")
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        ok, message = self.open(timeout=8.0)
        self.error = None if ok else message
        if ok:
            RUNTIME.update_status(camera_connected=True)
            print("✅ camera reconnected")
        else:
            time.sleep(2.0)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest

    def recent_frames(self) -> List[np.ndarray]:
        with self._lock:
            frames = list(self._history)
            if self._latest is not None:
                frames.append(self._latest)
        return frames[::-1]


def test_camera(camera_url: str):
    stream = CameraStream(camera_url)
    stream.publish = False
    ok, message = stream.open(timeout=10.0)
    frame = stream.read()
    stream.stop()

    if not ok or frame is None:
        return None, (
            f"## ❌ Camera connection failed\n\n`{mask_credentials(camera_url)}`\n\n"
            f"**Reason:** {message}\n\n"
            "1. Camera on the same network?  2. Host/port correct and RTSP enabled?\n"
            "3. Special characters in the password URL-encoded (`#` → `%23`)?\n"
            "4. Does the same URL open in VLC?"
        )

    return cv2.cvtColor(resize_frame(frame), cv2.COLOR_BGR2RGB), (
        f"## ✅ Camera connected\n\n`{mask_credentials(camera_url)}`\n\n"
        f"Resolution: **{frame.shape[1]} × {frame.shape[0]}**\n\nNow press **▶ Start detection**."
    )


# ============================================================================
# 7. MODELS
# ============================================================================

_PRESENCE_CACHE: Dict[str, Any] = {}
_CLASSIFIER_CACHE: Dict[str, Any] = {}

ABNORMAL_TOKENS = ("abnormal", "not-normal", "non-normal", "unhealthy", "non-healthy",
                   "not-healthy", "sick", "ill", "disease", "diseased", "infect",
                   "injur", "dead", "weak", "bad")
NORMAL_TOKENS = ("normal", "healthy", "health", "good", "fine", "well", "ok")


def label_group(class_name: str) -> Optional[str]:
    name = normalize_name(class_name)
    for token in ABNORMAL_TOKENS:
        if token in name:
            return "abnormal"
    for token in NORMAL_TOKENS:
        if token in name:
            return "normal"
    return None


def parse_class_override(text: str, names: Dict[int, str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not str(text or "").strip():
        return mapping
    lookup = {normalize_name(v): k for k, v in names.items()}
    for chunk in str(text).split(","):
        if "=" not in chunk:
            continue
        key, group = (part.strip() for part in chunk.split("=", 1))
        group = group.lower()
        if group not in ("normal", "abnormal", "ignore"):
            raise ValueError(f"override group must be normal/abnormal/ignore, got '{group}'")
        if key.isdigit() and int(key) in names:
            mapping[int(key)] = group
        elif normalize_name(key) in lookup:
            mapping[lookup[normalize_name(key)]] = group
        else:
            raise ValueError(f"override references unknown class '{key}'. Classes: {names}")
    return mapping


def load_presence_model(model_path: str):
    model_path = str(model_path).strip()
    if model_path in _PRESENCE_CACHE:
        return _PRESENCE_CACHE[model_path]
    print(f"\nLoading Stage-1 YOLO-World: {model_path}")
    model = YOLOWorld(model_path)
    model.set_classes(PRESENCE_CLASSES)
    _PRESENCE_CACHE[model_path] = model
    print(f"✅ Stage-1 ready — prompts: {PRESENCE_CLASSES}")
    return model


def load_classifier_model(model_path: str, override_text: str = ""):
    model_path = str(model_path).strip()
    cache_key = f"{model_path}||{override_text}"
    if cache_key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[cache_key]

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Stage-2 model not found:\n{model_path}")

    print(f"\nLoading Stage-2 model: {model_path}")
    model = YOLO(model_path)
    names = safe_model_names(model.names)
    task = getattr(model, "task", "unknown")
    print(f"  task={task}  classes={names}")

    if task not in ("classify", "detect"):
        raise ValueError(f"Stage-2 task '{task}' unsupported — use a classify or detect model.")

    groups = {cid: label_group(name) for cid, name in names.items()}
    groups = {cid: g for cid, g in groups.items() if g}
    groups.update(parse_class_override(override_text, names))

    abnormal_ids = sorted(cid for cid, g in groups.items() if g == "abnormal")
    normal_ids = sorted(cid for cid, g in groups.items() if g == "normal")

    if not abnormal_ids or not normal_ids:
        raise ValueError(
            "Could not map the model's classes to normal/abnormal.\n"
            f"Model classes: {names}\nDetected mapping: {groups}\n"
            "Use the 'class mapping override' box, e.g.  0=abnormal,1=normal")

    pack = {"model": model, "names": names, "task": task, "groups": groups,
            "abnormal_ids": abnormal_ids, "normal_ids": normal_ids}
    _CLASSIFIER_CACHE[cache_key] = pack
    print(f"✅ Stage-2 ready — abnormal={[names[i] for i in abnormal_ids]} "
          f"normal={[names[i] for i in normal_ids]}")
    return pack


# ============================================================================
# 8. STAGE 1 — presence
# ============================================================================


def detect_chicken_presence(presence_model, frame, confidence: float, min_area_ratio: float):
    with torch.inference_mode():
        results = presence_model.predict(source=frame, imgsz=MODEL_IMGSZ, conf=float(confidence),
                                         device=DEVICE, half=USE_HALF, max_det=30, verbose=False)
    result = results[0]
    annotated = frame.copy()
    detections: List[dict] = []

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False, detections, annotated

    names = safe_model_names(result.names)
    h, w = frame.shape[:2]
    frame_area = float(h * w)

    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].detach().cpu().tolist())
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = names.get(int(box.cls[0].item()), "")
        if not any(word in normalize_name(class_name)
                   for word in ("chicken", "hen", "rooster", "poultry")):
            continue

        area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
        if area_ratio < float(min_area_ratio):
            continue

        conf = float(box.conf[0].item())
        detections.append({"class_name": class_name, "confidence": conf,
                           "xyxy": (x1, y1, x2, y2), "area_ratio": area_ratio,
                           "score": conf * max(area_ratio, 1e-4)})

    detections.sort(key=lambda d: d["score"], reverse=True)

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, f"CHICKEN {det['confidence'] * 100:.0f}%", (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    return bool(detections), detections, annotated


def find_chicken_in_recent_frames(presence_model, frames, confidence, min_area_ratio):
    if not frames:
        return False, None, [], None, "no frames available"

    candidates = list(frames)[:PRESENCE_RECENT_FRAMES]
    best = {"found": False, "frame": candidates[0], "detections": [],
            "annotated": candidates[0].copy(), "score": 0.0}
    checked = 0

    for frame in candidates:
        checked += 1
        found, detections, annotated = detect_chicken_presence(
            presence_model, frame, confidence, min_area_ratio)
        if not found:
            continue
        score = max(d["score"] for d in detections)
        if score > best["score"]:
            best = {"found": True, "frame": frame, "detections": detections,
                    "annotated": annotated, "score": score}
        if max(d["confidence"] for d in detections) >= 0.55:
            break

    if best["found"]:
        top = max(d["confidence"] for d in best["detections"])
        return (True, best["frame"], best["detections"], best["annotated"],
                f"✅ chicken present — {len(best['detections'])} box(es), "
                f"best {top * 100:.1f}%, frames scanned {checked}")

    return (False, candidates[0], [], candidates[0].copy(),
            f"❌ no chicken/hen/rooster in {checked} frame(s)")


# ============================================================================
# 9. STAGE 2 — normal / abnormal
# ============================================================================


def classify_scene(classifier_pack: dict, frame: np.ndarray, min_confidence: float) -> dict:
    model = classifier_pack["model"]
    names = classifier_pack["names"]
    groups = classifier_pack["groups"]
    task = classifier_pack["task"]

    with torch.inference_mode():
        results = model.predict(source=frame, imgsz=MODEL_IMGSZ, device=DEVICE,
                                half=USE_HALF, verbose=False)
    result = results[0]

    if task == "classify":
        probs = getattr(result, "probs", None)
        if probs is None:
            return {"result": "UNDETECTED", "confidence": 0.0, "predicted_name": "-",
                    "details": "classifier returned no probabilities", "top": []}

        vector = probs.data.detach().cpu().tolist()
        abnormal_p = sum(vector[i] for i in classifier_pack["abnormal_ids"] if i < len(vector))
        normal_p = sum(vector[i] for i in classifier_pack["normal_ids"] if i < len(vector))
        top1_name = names.get(int(probs.top1), str(int(probs.top1)))
        ranked = sorted(enumerate(vector), key=lambda kv: kv[1], reverse=True)[:3]
        top = [f"{names.get(i, i)}={p * 100:.1f}%" for i, p in ranked]
        group, confidence = (("abnormal", abnormal_p) if abnormal_p >= normal_p
                             else ("normal", normal_p))
    else:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {"result": "UNDETECTED", "confidence": 0.0, "predicted_name": "-",
                    "details": "no boxes from stage-2 detection model", "top": []}
        best_conf, best_id = 0.0, None
        for box in boxes:
            cid, conf = int(box.cls[0].item()), float(box.conf[0].item())
            if groups.get(cid) in ("normal", "abnormal") and conf > best_conf:
                best_conf, best_id = conf, cid
        if best_id is None:
            return {"result": "UNDETECTED", "confidence": 0.0, "predicted_name": "-",
                    "details": "stage-2 boxes matched no mapped class", "top": []}
        group, confidence = groups[best_id], best_conf
        top1_name = names.get(best_id, str(best_id))
        top = [f"{top1_name}={confidence * 100:.1f}%"]

    if confidence < float(min_confidence):
        return {"result": "UNDETECTED", "confidence": confidence, "predicted_name": top1_name,
                "details": f"⚠️ low confidence — {top1_name} {confidence * 100:.1f}% "
                           f"(below {min_confidence * 100:.0f}%)", "top": top}

    final = "ABNORMAL" if group == "abnormal" else "NORMAL"
    return {"result": final, "confidence": confidence, "predicted_name": top1_name,
            "details": f"{final} — {top1_name} {confidence * 100:.1f}%  [{', '.join(top)}]",
            "top": top}


def _stamp(frame: np.ndarray, result: str, subtitle: str, count: int) -> None:
    colors = {"ABNORMAL": (0, 0, 255), "NORMAL": (0, 255, 0), "UNDETECTED": (0, 165, 255)}
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 78), (0, 0, 0), -1)
    cv2.putText(frame, f"FINAL: {result}  |  {subtitle}", (20, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, colors.get(result, (255, 255, 255)), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Chickens detected: {count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1, cv2.LINE_AA)


def run_two_stage_check(presence_model, classifier_pack, frames, presence_confidence,
                        classifier_min_confidence, min_area_ratio) -> dict:
    found, chosen, detections, annotated, presence_details = find_chicken_in_recent_frames(
        presence_model, frames, presence_confidence, min_area_ratio)

    if chosen is None:
        return {"result": "UNDETECTED", "chicken_count": 0, "confidence": 0.0,
                "presence_details": "no frames available", "classifier_details": "skipped",
                "annotated": None, "detections": []}

    if not found:
        _stamp(annotated, "UNDETECTED", "no chicken detected", 0)
        return {"result": "UNDETECTED", "chicken_count": 0, "confidence": 0.0,
                "presence_details": presence_details,
                "classifier_details": "skipped — Stage 1 found no chicken",
                "annotated": annotated, "detections": []}

    classification = classify_scene(classifier_pack, chosen, classifier_min_confidence)
    _stamp(annotated, classification["result"],
           f"{classification['predicted_name']} {classification['confidence'] * 100:.1f}%",
           len(detections))

    return {"result": classification["result"], "chicken_count": len(detections),
            "confidence": classification["confidence"], "presence_details": presence_details,
            "classifier_details": classification["details"],
            "annotated": annotated, "detections": detections}


# ============================================================================
# 10. LIVE OVERLAY
# ============================================================================


def draw_overlay(frame, elapsed, countdown, result, status, chicken_count, cooldown_left=0.0):
    if frame is None:
        frame = np.zeros((540, 960, 3), dtype=np.uint8)

    out = resize_frame(frame).copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, min(150, h)), (0, 0, 0), -1)

    for index, (text, color) in enumerate([
        (f"Camera time: {format_time(elapsed)}", (255, 255, 255)),
        (f"Next AI check in: {countdown:>2}s", (255, 255, 255)),
        (f"Chicken count: {chicken_count}", (0, 255, 255)),
    ]):
        cv2.putText(out, text, (20, 30 + index * 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.66, color, 2, cv2.LINE_AA)

    colors = {"ABNORMAL": (0, 0, 255), "NORMAL": (0, 255, 0), "UNDETECTED": (0, 165, 255),
              "CHECKING": (0, 255, 255), "WAITING": (255, 255, 255)}
    cv2.putText(out, f"Result: {result}", (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                colors.get(result, (255, 255, 255)), 2, cv2.LINE_AA)

    if w > 640:
        cv2.putText(out, str(status)[:52], (360, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (220, 220, 220), 1, cv2.LINE_AA)
        if cooldown_left > 0:
            cv2.putText(out, f"alert cooldown: {cooldown_left:.0f}s", (360, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 200, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ============================================================================
# 11. DETECTION LOOP
# ============================================================================

TIMELINE_COLUMNS = ["Time", "Chickens", "Result", "Confidence", "Stage 1 (presence)",
                    "Stage 2 (classifier)", "Alert"]


def start_detection(camera_url, presence_model_path, classifier_model_path, class_override,
                    output_folder, abnormal_api_url, normal_api_url, api_token,
                    presence_confidence, classifier_min_confidence, min_area_ratio,
                    cooldown_seconds):

    timeline: List[dict] = []
    gallery: List[str] = []
    timeline_df = pd.DataFrame(columns=TIMELINE_COLUMNS)

    try:
        presence_model = load_presence_model(presence_model_path)
        classifier_pack = load_classifier_model(classifier_model_path, class_override)
    except Exception as error:
        traceback.print_exc()
        yield None, f"## ❌ Model error\n\n```\n{error}\n```", timeline_df, []
        return

    local = api_base_url("127.0.0.1")
    abnormal_ep = build_endpoint("abnormal", abnormal_api_url, local + DEFAULT_ABNORMAL_PATH)
    normal_ep = build_endpoint("normal", normal_api_url, local + DEFAULT_NORMAL_PATH)

    output_folder = os.path.abspath(str(output_folder).strip())
    abnormal_folder = os.path.join(output_folder, ABNORMAL_SUBDIR)
    os.makedirs(abnormal_folder, exist_ok=True)
    RUNTIME.output_folder = output_folder

    dispatcher = AlertDispatcher(output_folder, token=str(api_token or "").strip(),
                                 cooldown=float(cooldown_seconds))

    def links_block() -> str:
        base = api_base_url(LAN_IP)
        return (
            "### Camera streaming API\n"
            f"- Live page: [{base}/view]({base}/view)\n"
            f"- MJPEG: `{base}/api/stream.mjpg`  ·  with AI boxes: `{base}/api/stream.mjpg?annotated=1`\n"
            f"- Snapshot: `{base}/api/snapshot.jpg`  ·  Status: `{base}/api/status`\n"
        )

    def api_block() -> str:
        summary = dispatcher.snapshot()
        return (
            "### Alert endpoints\n"
            f"- **ABNORMAL** → {abnormal_ep.status_text}\n"
            f"- **NORMAL** → {normal_ep.status_text}\n"
            f"- Cooldown `{float(cooldown_seconds):.0f}s` per type (state changes bypass)\n\n"
            f"sent `{summary['sent']}` · failed `{summary['failed']}` · "
            f"muted `{summary['skipped_cooldown']}` · unconfigured `{summary['skipped_unconfigured']}`\n\n"
            + links_block()
        )

    stream = CameraStream(camera_url)
    ok, message = stream.open(timeout=10.0)
    if not ok:
        dispatcher.shutdown()
        yield None, f"## ❌ Camera error\n\n`{mask_credentials(camera_url)}`\n\n{message}", timeline_df, []
        return
    stream.start()
    RUNTIME.update_status(running=True, camera_connected=True, result="WAITING")

    started = time.monotonic()
    next_check = started + CHECK_SECONDS
    check_number = 0
    last_result, last_count, last_confidence = "WAITING", 0, 0.0
    last_presence = last_classifier = "not checked yet"
    last_alert_note = "—"
    last_ui_push = 0.0
    pushed_rows = 0

    yield (draw_overlay(stream.read(), 0, int(CHECK_SECONDS), last_result, "CAMERA CONNECTED", 0),
           f"## ✅ Detector running\n\nChecking every **{CHECK_SECONDS:.0f}s**.\n\n{api_block()}",
           timeline_df, gallery)

    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            frame = stream.read()

            # ---------------------------------------------- AI CHECK ------
            if now >= next_check:
                check_number += 1
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                yield (draw_overlay(frame, elapsed, 0, "CHECKING", "STAGE 1: PRESENCE", last_count),
                       f"## ⏸ AI check #{check_number}\n\nStage 1 running…", gr_skip(), gr_skip())

                try:
                    check = run_two_stage_check(
                        presence_model, classifier_pack, stream.recent_frames(),
                        float(presence_confidence), float(classifier_min_confidence),
                        float(min_area_ratio))
                except Exception as error:
                    traceback.print_exc()
                    check = {"result": "UNDETECTED", "chicken_count": 0, "confidence": 0.0,
                             "presence_details": f"detection error: {error}",
                             "classifier_details": "skipped after error",
                             "annotated": frame.copy() if frame is not None else None,
                             "detections": []}

                result = check["result"]
                annotated = check["annotated"]
                RUNTIME.set_annotated(annotated)
                saved_path = None

                if result == "ABNORMAL" and annotated is not None:
                    saved_path = os.path.join(abnormal_folder,
                                              f"abnormal_{check_number:04d}_{stamp}.jpg")
                    cv2.imwrite(saved_path, annotated)
                    if os.path.exists(saved_path):
                        with open(saved_path, "rb") as fh:
                            save_image("chicken_abnormal", os.path.basename(saved_path), fh.read())
                    gallery.append(saved_path)
                    print(f"🔴 ABNORMAL saved → {saved_path}")

                if result in ("ABNORMAL", "NORMAL"):
                    endpoint = abnormal_ep if result == "ABNORMAL" else normal_ep
                    payload = build_payload(check_number, result, check["chicken_count"],
                                            check["presence_details"], check["classifier_details"],
                                            check["confidence"], camera_url, saved_path, annotated)
                    _, last_alert_note = dispatcher.submit(endpoint, result.lower(), payload)

                    # Tell the disease and predator services what the flock is
                    # doing. Best-effort: never blocks this loop, never fatal.
                    FARM_PEERS.publish(
                        kind=f"behaviour_{result.lower()}",
                        message=(f"Flock behaviour {result} on check "
                                 f"#{check_number} ({check['chicken_count']} birds, "
                                 f"{check['confidence']:.0%} confidence)."),
                        severity="warning" if result == "ABNORMAL" else "info",
                        data=payload,
                    )
                else:
                    last_alert_note = "no alert (undetected)"

                last_result = result
                last_count = check["chicken_count"]
                last_confidence = check["confidence"]
                last_presence = check["presence_details"]
                last_classifier = check["classifier_details"]

                RUNTIME.update_status(result=result, confidence=round(last_confidence, 4),
                                      chicken_count=last_count, check_number=check_number,
                                      last_check_at=now_iso(), last_alert=last_alert_note,
                                      presence_details=last_presence,
                                      classifier_details=last_classifier,
                                      camera_connected=stream.connected,
                                      reconnects=stream.reconnects)
                RUNTIME.add_event({"timestamp": now_iso(), "check_number": check_number,
                                   "result": result, "confidence": round(last_confidence, 4),
                                   "chicken_count": last_count, "alert": last_alert_note,
                                   "image_path": saved_path})

                timeline.append({"Time": datetime.now().strftime("%H:%M:%S"),
                                 "Chickens": last_count, "Result": result,
                                 "Confidence": f"{last_confidence * 100:.1f}%",
                                 "Stage 1 (presence)": last_presence,
                                 "Stage 2 (classifier)": last_classifier,
                                 "Alert": last_alert_note})
                timeline_df = pd.DataFrame(timeline, columns=TIMELINE_COLUMNS)
                pushed_rows = len(timeline)

                yield (draw_overlay(annotated, elapsed, int(CHECK_SECONDS), result, "CHECK COMPLETE",
                                    last_count, dispatcher.gate.remaining(result.lower())),
                       (f"## ✅ Check #{check_number} — **{result}**\n\n"
                        f"**Chickens:** `{last_count}` · **Confidence:** `{last_confidence * 100:.1f}%`\n\n"
                        f"**Stage 1:** {last_presence}\n\n**Stage 2:** {last_classifier}\n\n"
                        f"**Alert:** {last_alert_note}\n\n"
                        f"**Saved:** {'`' + saved_path + '`' if saved_path else 'not saved (only ABNORMAL)'}\n\n"
                        f"{api_block()}"),
                       timeline_df, gallery)

                while next_check <= time.monotonic():
                    next_check += CHECK_SECONDS
                continue

            # ------------------------------------------- LIVE PREVIEW -----
            if now - last_ui_push >= 1.0 / UI_FPS:
                last_ui_push = now
                countdown = max(0, int(round(next_check - now)))
                camera_state = "CCTV LIVE" if stream.connected else "RECONNECTING…"
                RUNTIME.update_status(next_check_in=countdown,
                                      camera_connected=stream.connected,
                                      reconnects=stream.reconnects)

                status = (
                    f"## ▶ {camera_state}\n\n"
                    f"**Camera:** `{mask_credentials(camera_url)}`  \n"
                    f"**Uptime:** `{format_time(elapsed)}` · **Next check:** `{countdown}s`  \n"
                    f"**Last result:** **{last_result}** (`{last_confidence * 100:.1f}%`) · "
                    f"**Chickens:** `{last_count}`  \n"
                    f"**Checks:** `{len(timeline)}` · **Abnormal saved:** `{len(gallery)}` · "
                    f"**Reconnects:** `{stream.reconnects}`\n\n"
                    f"**Stage 1:** {last_presence}  \n**Stage 2:** {last_classifier}  \n"
                    f"**Last alert:** {last_alert_note}\n\n{api_block()}"
                )

                display = draw_overlay(frame, elapsed, countdown, last_result, camera_state,
                                       last_count, dispatcher.gate.remaining(last_result.lower()))

                if len(timeline) != pushed_rows:
                    pushed_rows = len(timeline)
                    yield display, status, timeline_df, gallery
                else:
                    yield display, status, gr_skip(), gr_skip()

            time.sleep(0.005)

    finally:
        stream.stop()
        dispatcher.shutdown()
        RUNTIME.update_status(running=False, camera_connected=False)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("🛑 detector stopped, camera released.")


# ============================================================================
# 12. GRADIO UI
# ============================================================================

CUSTOM_CSS = """
.gradio-container { max-width: 1500px !important; }
#video_box img { object-fit: contain !important; background: #000 !important; border-radius: 14px !important; }
"""

blocks_kwargs: Dict[str, Any] = {"title": "Chicken CCTV — two-stage detector"}
if GRADIO_MAJOR < 6:
    blocks_kwargs["css"] = CUSTOM_CSS

LOCAL_BASE = api_base_url("127.0.0.1")
LAN_BASE = api_base_url(LAN_IP)

with gr.Blocks(**blocks_kwargs) as app:
    gr.Markdown(f"""
# 🐔 Chicken CCTV Detector — two-stage + streaming API

**Stage 1** YOLO-World finds `chicken / hen / rooster / poultry`.
**Stage 2** runs only if Stage 1 hits: your `.pt` classifies **NORMAL** / **ABNORMAL**.
Checks every **{CHECK_SECONDS:.0f}s**, alerts have a **{ALERT_COOLDOWN_SECONDS:.0f}s cooldown** per type,
only **ABNORMAL** frames are saved.

### 📡 Camera streaming API — reachable from any device on your network
| what | link |
|---|---|
| Live view page | [{LAN_BASE}/view]({LAN_BASE}/view) |
| MJPEG stream | `{LAN_BASE}/api/stream.mjpg` |
| MJPEG + AI boxes | `{LAN_BASE}/api/stream.mjpg?annotated=1` |
| Snapshot | `{LAN_BASE}/api/snapshot.jpg` |
| Status JSON | `{LAN_BASE}/api/status` |
| Events JSON | `{LAN_BASE}/api/events?limit=50` |
| API index | [{LAN_BASE}/]({LAN_BASE}/) |

The alert URL boxes below default to this machine's **built-in receiver**, so alerting
works immediately — check `{LAN_BASE}/api/received` to see them arrive. Replace them
with your own server URL whenever you have one.
""")

    with gr.Row():
        with gr.Column(scale=1):
            camera_url_input = gr.Textbox(value=CAMERA_URL, label="CCTV RTSP URL", lines=2)
            presence_model_input = gr.Textbox(value=PRESENCE_MODEL_PATH, label="Stage-1 YOLO-World model")
            classifier_model_input = gr.Textbox(value=CLASSIFIER_MODEL_PATH, label="Stage-2 .pt model")
            class_override_input = gr.Textbox(value="", label="Class mapping override (optional)",
                                              placeholder="0=abnormal,1=normal")
            output_input = gr.Textbox(value=OUTPUT_FOLDER, label="Output folder")

            gr.Markdown("#### Alert endpoints")
            abnormal_api_input = gr.Textbox(
                value=ABNORMAL_API_URL or (LOCAL_BASE + DEFAULT_ABNORMAL_PATH),
                label="ABNORMAL alert URL")
            normal_api_input = gr.Textbox(
                value=NORMAL_API_URL or (LOCAL_BASE + DEFAULT_NORMAL_PATH),
                label="NORMAL alert URL")
            api_token_input = gr.Textbox(value=API_AUTH_TOKEN, label="Bearer token (optional)",
                                         type="password")
            cooldown_slider = gr.Slider(0, 300, value=ALERT_COOLDOWN_SECONDS, step=5,
                                        label="Alert cooldown (seconds)")

            gr.Markdown("#### Thresholds")
            presence_conf_slider = gr.Slider(0.05, 0.60, value=PRESENCE_CONFIDENCE, step=0.01,
                                             label="Stage-1 chicken confidence")
            classifier_conf_slider = gr.Slider(0.10, 0.95, value=CLASSIFIER_MIN_CONFIDENCE, step=0.01,
                                               label="Stage-2 minimum confidence")
            min_area_slider = gr.Slider(0.0002, 0.0200, value=MIN_CHICKEN_BOX_AREA_RATIO, step=0.0002,
                                        label="Minimum box area ratio")

            test_btn = gr.Button("🔍 Test camera", variant="secondary")
            start_btn = gr.Button("▶ Start detection", variant="primary")
            stop_btn = gr.Button("■ Stop", variant="stop")

        with gr.Column(scale=2):
            video_output = gr.Image(label="CCTV / AI detection", elem_id="video_box",
                                    height=560, streaming=True)
            status_output = gr.Markdown("Waiting for camera…")

    with gr.Row():
        timeline_output = gr.Dataframe(value=pd.DataFrame(columns=TIMELINE_COLUMNS),
                                       headers=TIMELINE_COLUMNS, label="Detection timeline",
                                       interactive=False, wrap=True)
        gallery_output = gr.Gallery(label="Saved abnormal frames", columns=4,
                                    height="260px", object_fit="contain")

    test_btn.click(fn=test_camera, inputs=[camera_url_input], outputs=[video_output, status_output])

    run_event = start_btn.click(
        fn=start_detection,
        inputs=[camera_url_input, presence_model_input, classifier_model_input, class_override_input,
                output_input, abnormal_api_input, normal_api_input, api_token_input,
                presence_conf_slider, classifier_conf_slider, min_area_slider, cooldown_slider],
        outputs=[video_output, status_output, timeline_output, gallery_output],
        concurrency_limit=1,
    )

    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event])


# ============================================================================
# 13. LAUNCH
# ============================================================================


def find_free_port(start: int = 7860, end: int = 7900) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port between {start} and {end}")


def main() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    RUNTIME.output_folder = OUTPUT_FOLDER

    api_ok = API_SERVER.start()
    base_lan = api_base_url(LAN_IP)
    local = api_base_url("127.0.0.1")

    print("=" * 78)
    print("🐔 CHICKEN CCTV AI DETECTOR  (v2 — streaming API)")
    print("=" * 78)
    print("OS        :", platform.platform())
    print("Gradio    :", gr.__version__, "| Torch:", torch.__version__, "| Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU       :", torch.cuda.get_device_name(0))
    print("Camera    :", mask_credentials(CAMERA_URL))
    print("Stage 1   :", PRESENCE_MODEL_PATH)
    print("Stage 2   :", CLASSIFIER_MODEL_PATH,
          "(found)" if os.path.isfile(CLASSIFIER_MODEL_PATH) else "⚠️ NOT FOUND")
    print("Interval  :", f"{CHECK_SECONDS:.0f}s   Cooldown: {ALERT_COOLDOWN_SECONDS:.0f}s")
    print("Output    :", OUTPUT_FOLDER)
    print("-" * 78)

    if api_ok:
        print("📡 STREAMING API   :", base_lan)
        print("   live view page  :", f"{base_lan}/view")
        print("   MJPEG stream    :", f"{base_lan}/api/stream.mjpg")
        print("   MJPEG + AI boxes:", f"{base_lan}/api/stream.mjpg?annotated=1")
        print("   snapshot        :", f"{base_lan}/api/snapshot.jpg")
        print("   status JSON     :", f"{base_lan}/api/status")
        print("   events JSON     :", f"{base_lan}/api/events?limit=50")
        print("   received alerts :", f"{base_lan}/api/received")
        print("-" * 78)
        print("🔔 ALERT RECEIVERS (defaults, already working):")
        print("   abnormal POST   :", local + DEFAULT_ABNORMAL_PATH)
        print("   normal   POST   :", local + DEFAULT_NORMAL_PATH)
        print(build_endpoint("abnormal", ABNORMAL_API_URL, local + DEFAULT_ABNORMAL_PATH).status_text)
        print(build_endpoint("normal", NORMAL_API_URL, local + DEFAULT_NORMAL_PATH).status_text)
        if API_SERVER_HOST == "0.0.0.0":
            print("   (Windows Firewall may ask to allow Python on private networks — say yes,")
            print("    otherwise other devices cannot reach these links.)")
    else:
        print("⚠️ streaming API did not start:", API_SERVER.error)
    print("=" * 78)

    port = find_free_port()
    print(f"🖥  Gradio UI: http://127.0.0.1:{port}")

    app.queue()
    launch_kwargs: Dict[str, Any] = {"server_name": "127.0.0.1", "server_port": port,
                                     "share": False, "inbrowser": True, "show_error": True,
                                     # abnormal snapshots live under OUTPUT_FOLDER, outside cwd/temp
                                     "allowed_paths": [OUTPUT_FOLDER]}
    if GRADIO_MAJOR >= 6:
        launch_kwargs["css"] = CUSTOM_CSS

    try:
        app.launch(**launch_kwargs)
    except TypeError:
        launch_kwargs.pop("css", None)
        app.launch(**launch_kwargs)
    except KeyboardInterrupt:
        print("🛑 stopped by user")
    finally:
        API_SERVER.stop()


if __name__ == "__main__":
    main()
