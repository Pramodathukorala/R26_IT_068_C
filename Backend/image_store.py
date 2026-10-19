# ============================================================
# CAPTURED IMAGE STORE (local MongoDB)
# Shared by all three Farm Pulse backends. Only captured images
# are stored, in the "captured_images" collection of the local
# Poulty_plus database. Inserts run on a background thread, so a
# slow or stopped MongoDB never blocks detection or API requests.
#
# Env override: MONGO_URI
# ============================================================

import os
import queue
import threading
import time
from datetime import datetime, timezone

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/Poulty_plus")
COLLECTION = "captured_images"
RETRY_SECONDS = 30

_queue = queue.Queue(maxsize=200)
_worker = None
_worker_lock = threading.Lock()


def save_image(component, filename, data, content_type="image/jpeg"):
    """Queue one captured image for MongoDB. Never raises."""
    global _worker
    if not data:
        return
    with _worker_lock:
        if _worker is None:
            _worker = threading.Thread(target=_run, name="image-store", daemon=True)
            _worker.start()
    try:
        _queue.put_nowait({
            "component": component,
            "filename": filename,
            "content_type": content_type or "image/jpeg",
            "size_bytes": len(data),
            "image": bytes(data),
            "captured_at": datetime.now(timezone.utc),
        })
    except queue.Full:
        print(f"[image-store] queue full, dropped {filename}")


def _connect():
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    collection = client.get_default_database()[COLLECTION]
    collection.create_index([("component", 1), ("captured_at", -1)])
    print(f"[image-store] connected to {MONGO_URI} ({COLLECTION})")
    return collection


def _run():
    collection = None
    while True:
        doc = _queue.get()
        while True:
            try:
                if collection is None:
                    collection = _connect()
                collection.insert_one(doc)
                break
            except Exception as exc:
                collection = None
                print(f"[image-store] MongoDB unavailable ({type(exc).__name__}); retrying in {RETRY_SECONDS}s")
                time.sleep(RETRY_SECONDS)
