# ============================================================
# FARM SNAKE / SCORPION MONITOR - API SERVER (port 8002)
# EfficientNet-B3 whole-frame classifier from final.ipynb, serving the
# same routes the Farm Pulse app's predatorApi.js calls.
#
# Run:   python server.py                  (camera from STREAM_URL)
#        python server.py --source 0       (PC webcam)
#        python server.py --source D:\test.mp4
# Open:  http://127.0.0.1:8002/
#
# Env overrides: PREDATOR_RTSP_URL, PREDATOR_API_HOST, PREDATOR_API_PORT
# ============================================================

import os
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
)

import argparse
import csv
import json
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, time as dtime
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

try:
    import winsound  # Windows only
except ImportError:
    winsound = None

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

# Shared captured-image store lives one level up in Backend/
sys.path.insert(0, str(ROOT.parent))
from image_store import save_image
MODEL_CANDIDATES = [
    ROOT / "2nd best_efficientnet_b3.pth",
    Path(r"C:\Users\TechZiz.lk\Downloads\2nd best_efficientnet_b3.pth"),
]
STREAM_URL = os.getenv(
    "PREDATOR_RTSP_URL",
    "rtsp://admin:Aa20020607@192.168.8.102:554/cam/realmonitor?channel=1&subtype=1",
)
CLASS_NAMES = ["other", "scorpions", "snakes"]

OUTPUT_DIR = ROOT / "detection"
ALERT_DIR = OUTPUT_DIR / "alerts"
LOG_FILE = OUTPUT_DIR / "detections_log.csv"
FRONTEND = ROOT / "frontend" / "index.html"

# 0.0.0.0 so the phone on the same Wi-Fi can reach it; 8002 because the
# chicken API owns 8000 and the disease API owns 8001.
HOST = os.getenv("PREDATOR_API_HOST", "0.0.0.0")
PORT = int(os.getenv("PREDATOR_API_PORT", "8002"))
LOCAL_HOST = "127.0.0.1" if HOST == "0.0.0.0" else HOST

INPUT_SIZE = 300
INFERENCE_INTERVAL_SECONDS = 0.40
VOTE_WINDOW = 5
MIN_HITS_TO_CONFIRM = 3
ALERT_THRESHOLDS = {"snakes": 0.50, "scorpions": 0.50}
ALERT_COOLDOWN_SECONDS = 15
PLAY_BEEP_ON_ALERT = True

# Model labels -> species names used by the app and detection folders.
SPECIES = {"snakes": "snake", "scorpions": "scorpion"}
LABEL_OF = {v: k for k, v in SPECIES.items()}
CANONICAL = ("snake", "scorpion")
LOG_HEADER = ["timestamp", "species", "confidence", "similarity",
              "x1", "y1", "x2", "y2", "image_path"]
log_lock = threading.Lock()

# ============================================================
# MODEL (same loading logic as final.ipynb)
# ============================================================

def unpack_int4(obj):
    restored = {}
    for name, tensor in obj["packed_state"].items():
        meta = obj["metadata"][name]
        if not meta["quantized"]:
            restored[name] = tensor
            continue
        packed = tensor.to(torch.uint8).reshape(-1)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        q = torch.stack((low, high), dim=1).reshape(-1)
        q = q[:meta["numel"]].to(torch.float32) - 8.0
        restored[name] = (q * float(meta["scale"])).reshape(meta["shape"])
    return restored


def load_weights(path: Path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)

    names = None
    if isinstance(obj, dict) and obj.get("format") == "per_tensor_symmetric_int4_storage":
        state = unpack_int4(obj)
        names = obj.get("class_names")
    elif isinstance(obj, nn.Module):
        state = obj.state_dict()
    else:
        names = obj.get("class_names") or obj.get("classes")
        state = None
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            if key in obj and isinstance(obj[key], (dict, nn.Module)):
                state = obj[key]
                break
        if state is None:
            state = obj
        if isinstance(state, nn.Module):
            state = state.state_dict()

    target = "classifier.1.weight"
    match = [k for k in state if k.endswith(target)]
    if not match:
        raise KeyError(f"No '{target}' in {path.name}; not an EfficientNet-B3 state dict.")
    prefix = match[0][:-len(target)]
    if prefix:
        state = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}
    return state, names, (obj if isinstance(obj, dict) else {})


def canonical_label(label: str) -> str:
    v = str(label).strip().lower()
    if "scorpion" in v:
        return "scorpions"
    if "snake" in v:
        return "snakes"
    if "other" in v:
        return "other"
    return v


def canonical_species(name) -> str:
    """'snakes' / 'Snake' / 'scorpions' -> 'snake' / 'scorpion'; anything else unchanged."""
    label = canonical_label(name)
    return SPECIES.get(label, label)


def dead_bn_variances(state):
    """Count BatchNorm running_var entries that are zero (a sign of a bad export)."""
    return sum(int((v == 0).sum()) for k, v in state.items() if k.endswith("running_var"))


class Classifier:
    def __init__(self):
        self.enabled, self.error, self.dead_bn = False, None, 0
        self.lock = threading.Lock()  # camera thread and /api/detect share the model
        try:
            path = next((p for p in MODEL_CANDIDATES if p.is_file()), None)
            if path is None:
                raise FileNotFoundError("model .pth not found")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state, saved_names, meta = load_weights(path)
            class_names = list(saved_names) if saved_names else list(CLASS_NAMES)
            self.labels = [canonical_label(n) for n in class_names]
            self.dead_bn = dead_bn_variances(state)

            self.model = models.efficientnet_b3(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(class_names))
            self.model.load_state_dict(state, strict=True)
            self.model.eval().to(self.device)
            self.half = self.device.type == "cuda"
            if self.half:
                self.model.half()

            size = int(meta.get("input_size", INPUT_SIZE))
            self.preprocess = transforms.Compose([
                transforms.Resize(320, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(meta.get("imagenet_mean", [0.485, 0.456, 0.406]),
                                     meta.get("imagenet_std", [0.229, 0.224, 0.225])),
            ])
            self.enabled = True
            print(f"Model loaded: {path} | classes {self.labels} | device {self.device}")
        except Exception as e:
            self.error = str(e)
            print("Classifier disabled:", e)

    def predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        if self.half:
            t = t.half()
        with self.lock, torch.inference_mode():
            probs = torch.softmax(self.model(t).float(), dim=1)[0].cpu()
        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

# ============================================================
# DETECTION RECORDS (saved frame + CSV log row)
# ============================================================

def save_detection(frame, species, conf, probs=None):
    """Save a full frame into detection/<species>/ and log it. Returns the detection dict."""
    now = datetime.now()
    ts = now.isoformat(timespec="seconds")
    stamp = now.strftime("%Y%m%d_%H%M%S")
    ms = f"{now.microsecond // 1000:03d}"
    h, w = frame.shape[:2]

    img_path = OUTPUT_DIR / species / f"{species}_{stamp}_{ms}_conf{conf:.2f}.jpg"
    cv2.imwrite(str(img_path), frame)
    if img_path.exists():
        save_image(f"predator_{species}", img_path.name, img_path.read_bytes())

    with log_lock:
        new = not LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(LOG_HEADER)
            wr.writerow([ts, species, round(conf, 4), "", 0, 0, w, h, str(img_path)])

    return detection_dict(species, conf, w, h, ts, probs, str(img_path))


def detection_dict(species, conf, w, h, ts, probs=None, image_path=None):
    return {"species": species, "confidence": round(conf, 4), "box": [0, 0, w, h],
            "similarity": None,
            "classifier": {"label": species, "confidence": round(conf, 4)},
            "probabilities": {canonical_species(k): round(v, 4) for k, v in (probs or {}).items()},
            "reason": "", "image_path": image_path, "timestamp": ts}

# ============================================================
# ALERTS (per-species cooldown, history on disk)
# ============================================================

class AlertEngine:
    def __init__(self, cooldown=ALERT_COOLDOWN_SECONDS):
        self.cooldown = float(cooldown)
        self.last_fired = defaultdict(float)
        self.suppressed = defaultdict(int)
        self.lock = threading.Lock()
        self.history = deque(self._load_history(), maxlen=200)
        self.seq = max((int(a["id"].rsplit("_", 1)[-1]) for a in self.history
                        if str(a.get("id", "")).rsplit("_", 1)[-1].isdigit()), default=0)

    @staticmethod
    def _load_history():
        alerts = []
        for p in sorted(ALERT_DIR.glob("alert_*.json"), reverse=True)[:200]:
            try:
                alerts.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return alerts

    def cooldown_left(self, species):
        return max(0.0, self.cooldown - (time.time() - self.last_fired[species]))

    def trigger(self, species, detections=None, frame=None, source="camera", conf=None):
        """Fire an alert unless the species is inside its cooldown. Returns the alert or None."""
        now = time.time()
        with self.lock:
            if now - self.last_fired[species] < self.cooldown:
                self.suppressed[species] += 1
                return None
            self.last_fired[species] = now
            self.seq += 1
            seq = self.seq

        stamp = datetime.now()
        alert_id = f"{stamp.strftime('%Y%m%d_%H%M%S')}_{seq:04d}"

        snapshot = None
        if frame is not None:
            snap = frame.copy()
            text = f"ALERT: {species.upper()}" + (f" ({conf:.1%})" if conf is not None else "")
            cv2.putText(snap, text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2, cv2.LINE_AA)
            snapshot = str(ALERT_DIR / f"alert_{alert_id}.jpg")
            cv2.imwrite(snapshot, snap)

        alert = {"id": alert_id, "species": species,
                 "severity": "critical" if species == "both" else "high",
                 "message": f"{species.upper()} detected on camera.",
                 "timestamp": stamp.isoformat(timespec="seconds"),
                 "source": source, "cooldown_seconds": self.cooldown,
                 "detections": detections or [], "snapshot": snapshot}

        with self.lock:
            self.history.appendleft(alert)
        (ALERT_DIR / f"alert_{alert_id}.json").write_text(json.dumps(alert, indent=2), encoding="utf-8")
        print(f"!!! {species.upper()} ALERT ({source}): {snapshot or 'no frame'}")
        if PLAY_BEEP_ON_ALERT and winsound is not None:
            threading.Thread(target=winsound.Beep, args=(1000, 500), daemon=True).start()
        return alert

    def recent(self, n=25):
        with self.lock:
            return list(self.history)[:n]

# ============================================================
# MONITOR LOOP (camera -> classifier -> alerts)
# ============================================================

class Monitor:
    def __init__(self, source):
        self.source = source
        for d in (ALERT_DIR, OUTPUT_DIR / "snake", OUTPUT_DIR / "scorpion"):
            d.mkdir(parents=True, exist_ok=True)
        self.clf = Classifier()
        self.alerts = AlertEngine()
        self.lock = threading.Lock()
        self.jpeg = None
        self.raw = None
        self.connected = False
        self.stream_error = "connecting"
        self.reconnects = 0
        self.fps = 0.0
        self.frames = 0
        self.processed = 0
        self.unconfirmed = 0   # predator hits that never reached MIN_HITS_TO_CONFIRM
        self.current = []
        self.probs = {"other": 0.0, "scorpions": 0.0, "snakes": 0.0}
        threading.Thread(target=self._run, daemon=True).start()

    # ---------- helpers ----------
    def _set_frame(self, frame, raw=None):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
                if raw is not None:
                    self.raw = raw

    def _placeholder(self, text):
        img = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(img, text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        self._set_frame(img)

    def _open(self):
        src = self.source
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src, str) else cv2.VideoCapture(src)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def get_raw(self):
        with self.lock:
            return None if self.raw is None else self.raw.copy()

    # ---------- main loop ----------
    def _run(self):
        votes = deque(maxlen=VOTE_WINDOW)
        last_inf = 0.0
        probs = self.probs
        cap, failed = None, 0
        t0, frames = time.perf_counter(), 0

        while True:
            if cap is None or not cap.isOpened():
                self.connected = False
                self._placeholder("Connecting to camera...")
                cap = self._open()
                if not cap.isOpened():
                    print("Cannot open stream:", self.source)
                    self.stream_error = "cannot open camera stream"
                    self._placeholder("Camera offline - retrying")
                    cap = None
                    self.reconnects += 1
                    time.sleep(3)
                    continue
                failed = 0

            ok, frame = cap.read()
            if not ok or frame is None:
                failed += 1
                if failed >= 10:
                    print("Camera read failed repeatedly; reconnecting...")
                    self.stream_error = "camera read failed; reconnecting"
                    cap.release()
                    cap = None
                    time.sleep(1)
                continue
            failed = 0
            self.connected = True
            self.stream_error = None
            self.frames += 1
            frames += 1
            now = time.time()

            if self.clf.enabled and now - last_inf >= INFERENCE_INTERVAL_SECONDS:
                last_inf = now
                probs = self.clf.predict(frame)
                self.probs = probs
                self.processed += 1
                top = max(probs, key=probs.get)
                hit = top in ALERT_THRESHOLDS and probs[top] > ALERT_THRESHOLDS[top]
                votes.append(top if hit else "other")

                counts = Counter(votes)
                ts = datetime.now().isoformat(timespec="seconds")
                h, w = frame.shape[:2]
                current = []
                for label in ("snakes", "scorpions"):
                    if counts[label] < MIN_HITS_TO_CONFIRM:
                        continue
                    species, conf = SPECIES[label], probs[label]
                    det = detection_dict(species, conf, w, h, ts, probs)
                    if self.alerts.cooldown_left(species) == 0:
                        det = save_detection(frame, species, conf, probs)
                        self.alerts.trigger(species, detections=[det], frame=frame, conf=conf)
                    else:
                        self.alerts.trigger(species)  # counts as suppressed
                    current.append(det)
                if hit and not current:
                    self.unconfirmed += 1
                with self.lock:
                    self.current = current

            display = frame.copy()
            color = (0, 0, 255) if self.current else (0, 220, 0)
            cv2.putText(display, f"Snake {probs['snakes']:.0%}  Scorpion {probs['scorpions']:.0%}  "
                        f"Other {probs['other']:.0%}", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            self._set_frame(display, raw=frame)

            elapsed = time.perf_counter() - t0
            if elapsed >= 1.0:
                self.fps = round(frames / elapsed, 1)
                t0, frames = time.perf_counter(), 0

    # ---------- API data ----------
    def status(self):
        def count(species):
            return sum(1 for _ in (OUTPUT_DIR / species).glob(f"{species}_*.jpg"))
        with self.lock:
            current = list(self.current)
        latest = self.alerts.recent(1)
        return {
            "stream_connected": self.connected,
            "stream_error": None if self.connected else self.stream_error,
            "reconnects": self.reconnects,
            "fps": self.fps if self.connected else 0,
            "frames": self.frames,
            "processed": self.processed,
            "rejected_boxes": self.unconfirmed,
            "saved": {"snake": count("snake"), "scorpion": count("scorpion")},
            "mode": "efficientnet-b3",
            "probabilities": {canonical_species(k): round(v, 4) for k, v in self.probs.items()},
            "classifier": {"enabled": self.clf.enabled, "error": self.clf.error,
                           "threshold": ALERT_THRESHOLDS["snakes"],
                           "vote": f"{MIN_HITS_TO_CONFIRM}/{VOTE_WINDOW}",
                           "dead_bn_variances": self.clf.dead_bn},
            "conf_threshold": ALERT_THRESHOLDS["snakes"],
            "openset": None,
            "cooldown": {sp: {"left": round(self.alerts.cooldown_left(sp), 2),
                              "total": self.alerts.cooldown,
                              "suppressed": self.alerts.suppressed[sp]} for sp in CANONICAL},
            "current": current,
            "latest_alert": latest[0] if latest else None,
        }

    def detect_image(self, img, want=None, save=True):
        """Single uploaded photo -> same shape the YOLO backend returned."""
        if not self.clf.enabled:
            raise RuntimeError(f"classifier disabled: {self.clf.error}")
        probs = self.clf.predict(img)
        h, w = img.shape[:2]
        ts = datetime.now().isoformat(timespec="seconds")
        targets = [LABEL_OF[want]] if want else ["snakes", "scorpions"]
        top = max(probs, key=probs.get)

        accepted, rejected = [], []
        for label in targets:
            species, conf = SPECIES[label], probs[label]
            if (want or label == top) and conf > ALERT_THRESHOLDS[label]:
                accepted.append(save_detection(img, species, conf, probs) if save
                                else detection_dict(species, conf, w, h, ts, probs))
            elif conf > 0.05:
                d = detection_dict(species, conf, w, h, ts, probs)
                d["reason"] = (f"confidence {conf:.0%} <= threshold {ALERT_THRESHOLDS[label]:.0%}"
                               if conf <= ALERT_THRESHOLDS[label] else f"top class is {canonical_species(top)}")
                rejected.append(d)

        return {"found": bool(accepted), "species_requested": want or "any",
                "count": len(accepted), "detections": accepted, "filtered_out": rejected,
                "top_class": canonical_species(top),
                "probabilities": {canonical_species(k): round(v, 4) for k, v in probs.items()},
                "note": "whole-image EfficientNet-B3 classification; box is the full frame"}

# ============================================================
# STATS (from detections_log.csv)
# ============================================================

COUNT_NOTE = ("A count is one confirmed detection frame. Each alert saves one frame, "
              f"so a single animal in view logs at most one count per {ALERT_COOLDOWN_SECONDS}s.")


def load_log(since=None, until=None):
    rows = []
    if not LOG_FILE.exists():
        return rows
    with log_lock, open(LOG_FILE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(r["timestamp"])
            except (ValueError, KeyError, TypeError):
                continue
            sp = canonical_species(r.get("species", ""))
            if sp not in CANONICAL or (since and ts < since) or (until and ts >= until):
                continue
            rows.append((ts, sp))
    return rows


def empty_buckets(labels):
    return [{"label": l, "snake": 0, "scorpion": 0} for l in labels]


def stats_hourly(date=None):
    day = datetime.fromisoformat(date).date() if date else datetime.now().date()
    start = datetime.combine(day, dtime.min)
    buckets = empty_buckets([f"{h:02d}" for h in range(24)])
    for ts, sp in load_log(start, start + timedelta(days=1)):
        buckets[ts.hour][sp] += 1
    totals = {sp: sum(b[sp] for b in buckets) for sp in CANONICAL}
    return {"mode": "hourly", "date": day.isoformat(), "unit": "hour", "buckets": buckets,
            "totals": totals, "total": sum(totals.values()), "note": COUNT_NOTE}


def stats_daily(days=7):
    days = max(1, min(90, days))
    today = datetime.now().date()
    day_list = [today - timedelta(days=days - 1 - i) for i in range(days)]
    index = {d: i for i, d in enumerate(day_list)}
    buckets = empty_buckets([d.strftime("%d %b") for d in day_list])
    for ts, sp in load_log(datetime.combine(day_list[0], dtime.min)):
        i = index.get(ts.date())
        if i is not None:
            buckets[i][sp] += 1
    totals = {sp: sum(b[sp] for b in buckets) for sp in CANONICAL}
    return {"mode": "daily", "days": days, "unit": "day",
            "from": day_list[0].isoformat(), "to": day_list[-1].isoformat(),
            "buckets": buckets, "totals": totals, "total": sum(totals.values()), "note": COUNT_NOTE}


def stats_summary():
    now = datetime.now()
    today_start = datetime.combine(now.date(), dtime.min)

    def tally(rows):
        out = {sp: 0 for sp in CANONICAL}
        for _, sp in rows:
            out[sp] += 1
        return out

    rows = load_log()
    return {"date": now.date().isoformat(),
            "today": tally(r for r in rows if r[0] >= today_start),
            "last_hour": tally(r for r in rows if r[0] >= now - timedelta(hours=1)),
            "all_time": tally(rows),
            "first_logged": rows[0][0].isoformat() if rows else None,
            "last_logged": rows[-1][0].isoformat() if rows else None,
            "note": COUNT_NOTE}

# ============================================================
# HTTP SERVER
# ============================================================

class HttpError(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code, self.detail = code, detail


def species_or_404(name):
    sp = canonical_species(unquote(name))
    if sp not in CANONICAL:
        raise HttpError(404, "species must be snake or scorpion")
    return sp


class Handler(BaseHTTPRequestHandler):
    monitor: Monitor = None

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _dispatch(self, fn):
        try:
            fn()
        except HttpError as e:
            self._json({"detail": e.detail}, e.code)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            print("Request failed:", self.path, e)
            self._json({"detail": str(e)}, 500)

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        self._dispatch(self._get)

    def do_POST(self):
        self._dispatch(self._post)

    # ---------- GET ----------
    def _get(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        arg = lambda k, d: q.get(k, [d])[0]
        path = url.path.rstrip("/") or "/"
        parts = path.strip("/").split("/")
        m = self.monitor

        if path in ("/", "/index.html"):
            if FRONTEND.is_file():
                self._send(200, FRONTEND.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(200, b"Snake & Scorpion API - see /api/status", "text/plain")
        elif path == "/api/status":
            self._json(m.status())
        elif path == "/api/alerts":
            limit = int(arg("limit", "25"))
            self._json({"count": len(m.alerts.history), "suppressed": dict(m.alerts.suppressed),
                        "alerts": m.alerts.recent(limit)})
        elif path == "/api/alert":
            r = m.alerts.recent(1)
            self._json({"alert": r[0] if r else None,
                        "cooldown": {sp: round(m.alerts.cooldown_left(sp), 2) for sp in CANONICAL}})
        elif path == "/api/stats/summary":
            self._json(stats_summary())
        elif path == "/api/stats/daily":
            self._json(stats_daily(int(arg("days", "7"))))
        elif path == "/api/stats/hourly":
            try:
                self._json(stats_hourly(arg("date", None)))
            except ValueError:
                raise HttpError(400, "date must be YYYY-MM-DD")
        elif len(parts) == 3 and parts[:2] == ["api", "captures"]:
            sp = species_or_404(parts[2])
            d = OUTPUT_DIR / sp
            files = sorted(d.glob(f"{sp}_*.jpg"), key=lambda p: p.stat().st_mtime,
                           reverse=True)[:int(arg("limit", "6"))]
            self._json({"species": sp, "folder": str(d),
                        "files": [{"name": p.name, "url": f"/captures/{sp}/{p.name}"} for p in files]})
        elif len(parts) == 3 and parts[0] == "captures":
            sp = species_or_404(parts[1])
            name = Path(unquote(parts[2])).name   # basename only - no path traversal
            p = OUTPUT_DIR / sp / name
            if not p.is_file():
                raise HttpError(404, "not found")
            self._send(200, p.read_bytes(), "image/jpeg")
        elif path == "/video_feed":
            self._stream()
        else:
            raise HttpError(404, "not found")

    # ---------- POST ----------
    def _post(self):
        url = urlparse(self.path)
        path = url.path.rstrip("/")
        q = parse_qs(url.query)
        m = self.monitor

        if path in ("/api/detect", "/api/detect/snake", "/api/detect/scorpion"):
            want = path.rsplit("/", 1)[-1] if path != "/api/detect" else None
            save = q.get("save", ["true"])[0].lower() not in ("0", "false", "no")
            img = self._upload_image()
            try:
                self._json(m.detect_image(img, want, save))
            except RuntimeError as e:
                raise HttpError(503, str(e))
        elif path == "/api/alert":
            try:
                payload = json.loads(self._body() or b"{}")
            except json.JSONDecodeError:
                raise HttpError(400, "body must be JSON")
            species = payload.get("species", "snake")
            species = species if species == "both" else canonical_species(species)
            if species not in CANONICAL and species != "both":
                raise HttpError(400, "species must be snake, scorpion or both")
            a = m.alerts.trigger(species, detections=payload.get("detections"),
                                 frame=m.get_raw(), source=payload.get("source", "external"))
            if a is None:
                self._json({"fired": False, "reason": "within cooldown",
                            "cooldown_left": round(m.alerts.cooldown_left(species), 2)})
            else:
                self._json({"fired": True, "alert": a})
        else:
            raise HttpError(404, "not found")

    def _upload_image(self):
        """Pull the 'file' part out of a multipart/form-data body and decode it."""
        ctype = self.headers.get("Content-Type", "")
        body = self._body()
        if not body:
            raise HttpError(400, "no file uploaded")
        data = body
        if ctype.startswith("multipart/form-data"):
            msg = BytesParser(policy=email_policy).parsebytes(
                b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
            parts = [p for p in msg.iter_parts() if p.get_param("name", header="content-disposition") == "file"]
            if not parts:
                parts = [p for p in msg.iter_parts() if p.get_filename()]
            if not parts:
                raise HttpError(400, "multipart body has no 'file' field")
            data = parts[0].get_payload(decode=True) or b""
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HttpError(400, "Could not decode that file as an image.")
        return img

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                with self.monitor.lock:
                    jpg = self.monitor.jpeg
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=STREAM_URL,
                    help="camera URL, video file, or webcam index (e.g. 0)")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    Handler.monitor = Monitor(source)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    local = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Dashboard: http://{local}:{args.port}/  (listening on {args.host}:{args.port})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
