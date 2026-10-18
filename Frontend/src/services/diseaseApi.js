import { DISEASE_API_BASE_URL } from '../config/api';

// Inference on a CPU-bound Keras model plus Grad-CAM takes a few seconds, and
// the upload itself travels over the LAN — so this is deliberately generous.
const PREDICT_TIMEOUT_MS = 45000;
const HEALTH_TIMEOUT_MS = 5000;

export const SCAN_TYPE = { FECAL: 'fecal', BODY: 'body' };

// Error codes raised by the FastAPI backend (app/main.py, app/services/
// input_validator.py). Kept here so the screen can branch on them by name.
export const ERROR_CODE = {
  INVALID_IMAGE_FILE: 'INVALID_IMAGE_FILE',
  IMAGE_QUALITY_FAILED: 'IMAGE_QUALITY_FAILED',
  UNSUPPORTED_IMAGE_TYPE: 'UNSUPPORTED_IMAGE_TYPE',
  WRONG_IMAGE_TYPE: 'WRONG_IMAGE_TYPE',
  LOW_CONFIDENCE: 'LOW_CONFIDENCE_OR_UNSUPPORTED_IMAGE',
  NETWORK: 'NETWORK_UNREACHABLE',
  TIMEOUT: 'REQUEST_TIMEOUT',
  SERVER: 'SERVER_ERROR',
};

/**
 * Error carrying the backend's structured `detail` payload.
 * `code` is one of ERROR_CODE; `detail` holds whatever extra fields the
 * backend attached (probabilities, image_quality, detected_type, ...).
 */
export class DiseaseApiError extends Error {
  constructor(code, message, detail = null, action = null) {
    super(message);
    this.name = 'DiseaseApiError';
    this.code = code;
    this.detail = detail;
    this.action = action;
  }
}

/**
 * Prefer this over `instanceof`: it still identifies the error correctly if
 * Babel ever transpiles the class down for a non-Hermes target.
 */
export function isDiseaseApiError(error) {
  return Boolean(error) && error.name === 'DiseaseApiError';
}

function normalizeErrorBody(body, httpStatus) {
  // FastAPI puts our dict under `detail`; HTTPException(detail="...") puts a
  // plain string there instead.
  const detail = body && body.detail;

  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    return new DiseaseApiError(
      detail.error_code || ERROR_CODE.SERVER,
      detail.message || 'The image could not be analyzed.',
      detail,
      detail.action || null,
    );
  }

  if (typeof detail === 'string' && detail.length > 0) {
    return new DiseaseApiError(ERROR_CODE.SERVER, detail, null, null);
  }

  return new DiseaseApiError(
    ERROR_CODE.SERVER,
    `The AI backend returned an unexpected response (HTTP ${httpStatus}).`,
  );
}

/**
 * Upload one image and get a diagnosis back.
 *
 * @param {'fecal'|'body'} scanType  which model to route to
 * @param {{uri: string, fileName?: string, type?: string}} asset  picked image
 * @returns {Promise<object>} the PredictionResponse from the backend
 */
export async function analyzeImage(scanType, asset) {
  if (!asset || !asset.uri) {
    throw new DiseaseApiError(
      ERROR_CODE.INVALID_IMAGE_FILE,
      'No image selected.',
    );
  }

  const endpoint =
    scanType === SCAN_TYPE.FECAL ? '/predict/fecal' : '/predict/body';

  const formData = new FormData();
  formData.append('file', {
    uri: asset.uri,
    // The backend reads bytes and lets Pillow sniff the format, but a sane
    // filename/type keeps python-multipart and the server logs readable.
    name: asset.fileName || `scan_${Date.now()}.jpg`,
    type: asset.type || 'image/jpeg',
  });

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PREDICT_TIMEOUT_MS);

  let response;
  try {
    response = await fetch(`${DISEASE_API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
      // Content-Type is intentionally omitted: fetch must set the multipart
      // boundary itself.
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new DiseaseApiError(
        ERROR_CODE.TIMEOUT,
        'The AI backend took too long to respond.',
        null,
        'Check that the server is still running, then try again.',
      );
    }
    throw new DiseaseApiError(
      ERROR_CODE.NETWORK,
      'Unable to reach the AI backend.',
      null,
      `Confirm the server is running and that DISEASE_API_BASE_URL (${DISEASE_API_BASE_URL}) matches the address it printed.`,
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
    throw normalizeErrorBody(body, response.status);
  }

  return body;
}

/** Turn the backend's relative `/gradcam/xxx.jpg` path into a loadable URL. */
export function gradcamImageUrl(relativePath) {
  if (!relativePath) return null;
  if (/^https?:\/\//i.test(relativePath)) return relativePath;
  return `${DISEASE_API_BASE_URL}${relativePath}`;
}

/** Cheap reachability probe for the status pill on the scanner screen. */
export async function checkDiseaseApiHealth() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
  try {
    const response = await fetch(`${DISEASE_API_BASE_URL}/health`, {
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}
