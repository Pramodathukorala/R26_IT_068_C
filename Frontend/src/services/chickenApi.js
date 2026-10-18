import { CHICKEN_API_BASE_URL } from '../config/api';

// The chicken detector is not FastAPI - it is the plain ThreadingHTTPServer
// defined in Final.ipynb (class ApiHandler). Its routes are:
//
//   GET /health                 {"ok": true, ...}
//   GET /api/status             latest AI result
//   GET /api/events?limit=      recent checks (image_url when a frame was saved)
//   GET /api/abnormal?limit=    saved abnormal frames, newest first
//   GET /api/images/<file>      one saved abnormal frame (JPEG)
//   GET /api/received?limit=    alerts pushed in by other services
//   GET /api/feed?limit=        own events + peers', merged
//   GET /api/feed/peers         per-peer delivery counters
//   GET /api/snapshot.jpg       single frame  (?annotated=1 for boxes)
//   GET /api/stream.mjpg        MJPEG stream  (?annotated=1 for boxes)
//   GET /view, GET /            HTML pages
//   POST /api/alerts/abnormal, /api/alerts/normal, /api/alerts

const REQUEST_TIMEOUT_MS = 6000;

async function getJson(path) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${CHICKEN_API_BASE_URL}${path}`, {
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

export function fetchChickenStatus() {
  return getJson('/api/status');
}

export function fetchChickenEvents(limit = 20) {
  return getJson(`/api/events?limit=${limit}`);
}

/**
 * The cross-service feed: this detector's own checks merged with whatever the
 * disease and predator services pushed to /api/alerts/*. Empty `received` just
 * means no peer has pushed anything yet.
 */
export function fetchChickenFeed(limit = 50) {
  return getJson(`/api/feed?limit=${limit}`);
}

export function chickenStreamUrl(annotated = true) {
  return `${CHICKEN_API_BASE_URL}/api/stream.mjpg${annotated ? '?annotated=1' : ''}`;
}

export function chickenSnapshotUrl(annotated = true) {
  return `${CHICKEN_API_BASE_URL}/api/snapshot.jpg${annotated ? '?annotated=1' : ''}`;
}

/** Cheap reachability probe for the dashboard's health roll-up. */
export function checkChickenApiHealth() {
  return getJson('/health');
}

// ─── Saved abnormal frames ────────────────────────────────────────────────────
// Every ABNORMAL check writes an annotated JPEG on the backend. Those files are
// exposed read-only as:
//
//   GET /api/abnormal?limit=   newest-first history rows (metadata + filename)
//   GET /api/images            the same files, listed
//   GET /api/images/<file>     one JPEG
//
// A row carries `image_url` already built by the backend from its own LAN IP.
// We deliberately ignore that and rebuild the URL from CHICKEN_API_BASE_URL, so
// the images keep loading over an `adb reverse` tunnel (127.0.0.1) and after the
// PC's LAN address changes.

export function fetchAbnormalHistory(limit = 12) {
  return getJson(`/api/abnormal?limit=${limit}`);
}

export function abnormalImageUrl(filename) {
  if (!filename) return null;
  return `${CHICKEN_API_BASE_URL}/api/images/${encodeURIComponent(filename)}`;
}
