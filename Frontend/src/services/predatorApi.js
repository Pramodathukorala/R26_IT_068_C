import { PREDATOR_API_BASE_URL } from '../config/api';

// Every path in this file was read off the running service's own OpenAPI
// document (GET /openapi.json), so it matches the notebook cell that defines
// the routes. The backend exposes exactly these:
//
//   POST /api/detect, /api/detect/snake, /api/detect/scorpion
//   GET  /api/alert            latest alert + per-species cooldown
//   POST /api/alert            raise one by hand (cooldown still applies)
//   GET  /api/alerts           alert history
//   GET  /api/status           stream/worker state, cooldowns, latest alert
//   GET  /api/captures/{species}, GET /captures/{species}/{name}
//   GET  /api/stats/hourly, /api/stats/daily, /api/stats/summary
//   GET  /video_feed           MJPEG with boxes drawn
//   GET  /                     HTML dashboard
//
// There is no /health, no /api/detections and no /api/camera/* - do not add
// calls to those, they 404.

// Polling calls are small JSON reads over the LAN, so they fail fast.
const REQUEST_TIMEOUT_MS = 6000;
// A single-image YOLO pass on CPU takes a couple of seconds, plus the upload.
const DETECT_TIMEOUT_MS = 45000;

export const SPECIES = { SNAKE: 'snake', SCORPION: 'scorpion' };

export class PredatorApiError extends Error {
  constructor(message, cause = null) {
    super(message);
    this.name = 'PredatorApiError';
    this.cause = cause;
  }
}

export function isPredatorApiError(error) {
  return Boolean(error) && error.name === 'PredatorApiError';
}

async function request(path, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response;
  try {
    response = await fetch(`${PREDATOR_API_BASE_URL}${path}`, {
      headers: { Accept: 'application/json', ...(options.headers || {}) },
      ...options,
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new PredatorApiError('The detection backend took too long to respond.', error);
    }
    throw new PredatorApiError(
      `Unable to reach the detection backend at ${PREDATOR_API_BASE_URL}.`,
      error,
    );
  } finally {
    clearTimeout(timer);
  }

  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = null;
  }

  if (!response.ok) {
    const detail = body && body.detail;
    throw new PredatorApiError(
      typeof detail === 'string' && detail
        ? detail
        : `The detection backend returned HTTP ${response.status}.`,
    );
  }

  return body;
}

/** Stream state, fps, per-species cooldowns and the latest alert. */
export function fetchPredatorStatus() {
  return request('/api/status');
}

/**
 * Today / last hour / all-time counts out of detections_log.csv.
 *
 * The raw response carries no `total` and names the newest row `last_logged`;
 * both are filled in here so screens do not have to know that.
 */
export async function fetchPredatorSummary() {
  const data = await request('/api/stats/summary');
  const withTotal = tally => ({
    snake: (tally && tally.snake) || 0,
    scorpion: (tally && tally.scorpion) || 0,
    total: ((tally && tally.snake) || 0) + ((tally && tally.scorpion) || 0),
  });
  return {
    ...data,
    today: withTotal(data.today),
    last_hour: withTotal(data.last_hour),
    all_time: withTotal(data.all_time),
    last_detection: data.last_logged || null,
  };
}

/**
 * Alerts that actually fired, i.e. survived the 15 s per-species cooldown.
 *
 * Also flattened into log rows: an alert carries the boxes that caused it
 * under `detections`, so confidence and open-set similarity come from the
 * first of those. A hand-raised alert (source `mobile-panic-button`) has none,
 * and its confidence stays null - the UI renders that as a dash.
 */
export async function fetchPredatorAlerts(limit = 12) {
  const data = await request(`/api/alerts?limit=${limit}`);
  const alerts = (data && data.alerts) || [];

  return {
    count: (data && data.count) || 0,
    suppressed: (data && data.suppressed) || {},
    alerts,
    detections: alerts.map(alert => {
      const first = (alert.detections && alert.detections[0]) || null;
      return {
        id: alert.id,
        species: alert.species,
        severity: alert.severity,
        message: alert.message,
        source: alert.source,
        timestamp: alert.timestamp,
        confidence: first ? first.confidence : null,
        similarity: first ? first.similarity : null,
        // Absolute Windows path; predatorImageUrl turns it into a URL.
        imagePath: first ? first.image_path : null,
      };
    }),
  };
}

/** One bucket per day, oldest first - feeds the trend chart. */
export function fetchPredatorDaily(days = 7) {
  return request(`/api/stats/daily?days=${days}`);
}

/** 24 hourly buckets for one day (defaults to today). */
export function fetchPredatorHourly(date) {
  return request(`/api/stats/hourly${date ? `?date=${date}` : ''}`);
}

/** The most recent saved crops for one species, as servable URLs. */
export function fetchPredatorCaptures(species, limit = 6) {
  return request(`/api/captures/${species}?limit=${limit}`);
}

/** Raise an alert by hand - the same cooldown rules apply. */
export function pushPredatorAlert(species = 'both', source = 'mobile-app') {
  return request('/api/alert', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ species, source }),
  });
}

/**
 * Upload one photo and run the full filter stack on it.
 *
 * @param {{uri: string, fileName?: string, type?: string}} asset picked image
 * @param {'snake'|'scorpion'|null} want restrict to one species
 */
export async function scanImage(asset, want = null) {
  if (!asset || !asset.uri) {
    throw new PredatorApiError('No image selected.');
  }

  const endpoint = want ? `/api/detect/${want}` : '/api/detect';

  const formData = new FormData();
  formData.append('file', {
    uri: asset.uri,
    name: asset.fileName || `scan_${Date.now()}.jpg`,
    type: asset.type || 'image/jpeg',
  });

  return request(
    endpoint,
    {
      method: 'POST',
      body: formData,
      // Content-Type is intentionally omitted: fetch must set the multipart
      // boundary itself.
    },
    DETECT_TIMEOUT_MS,
  );
}

/** MJPEG feed with the backend's bounding boxes already drawn on it. */
export function predatorStreamUrl(nonce = 0) {
  return `${PREDATOR_API_BASE_URL}/video_feed${nonce ? `?n=${nonce}` : ''}`;
}

/**
 * Turn an image reference from the API into something <Image> can load.
 *
 * The detector reports saved frames as absolute Windows paths inside
 * detection/snake or detection/scorpion. Those two folders are exactly what
 * /captures/{species}/{name} serves, so the basename is all that is needed.
 * Anything under detection/alerts has no route, so it returns null rather than
 * a URL that would 404.
 */
export function predatorImageUrl(pathOrUrl) {
  if (!pathOrUrl) return null;
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  if (pathOrUrl.startsWith('/')) return `${PREDATOR_API_BASE_URL}${pathOrUrl}`;

  const name = String(pathOrUrl).split(/[\\/]/).pop();
  if (!name) return null;

  const species = name.startsWith('scorpion')
    ? 'scorpion'
    : name.startsWith('snake')
    ? 'snake'
    : null;
  if (!species) return null; // e.g. alert_*.jpg - no route serves those

  return `${PREDATOR_API_BASE_URL}/captures/${species}/${name}`;
}

/**
 * Cheap reachability probe. The service has no /health, so /api/status is the
 * lightest thing that proves it is up.
 */
export function checkPredatorApiHealth() {
  return request('/api/status');
}
