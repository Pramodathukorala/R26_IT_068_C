// The danger-alarm rule for the predator (snake & scorpion) module.
//
// Deliberately NOT the chicken module's rule: [[alarmState]] there waits for
// two consecutive abnormal checks before it sounds, because one odd frame of
// chicken behaviour is not worth waking anybody for. A snake in the coop is.
// So there is no streak here — one confirmed snake/scorpion detection fires the
// alarm immediately.
//
// Kept pure and free of React (and of the API layer) so the rule can be
// reasoned about on its own.

/** Species the backend can report that count as a danger. */
export const DANGER_SPECIES = ['snake', 'scorpion', 'both'];

/** Pretty names for the alert card. `both` is what the panic button raises. */
const SPECIES_LABEL = { snake: 'Snake', scorpion: 'Scorpion', both: 'Snake & Scorpion' };

const SPECIES_EMOJI = { snake: '🐍', scorpion: '🦂', both: '⚠️' };

/**
 * The state a freshly mounted screen starts from.
 *
 * `armed` is false until the first status response has been folded in — see
 * nextPredatorAlarmState for why that matters.
 */
export const IDLE_PREDATOR_ALARM_STATE = { seenAlertId: null, armed: false };

/** Is this something the alarm should sound for? */
export function isDangerSpecies(species) {
  return DANGER_SPECIES.includes(String(species || '').toLowerCase());
}

export function speciesLabel(species) {
  const key = String(species || '').toLowerCase();
  return SPECIES_LABEL[key] || (key ? key[0].toUpperCase() + key.slice(1) : 'Unknown');
}

export function speciesEmoji(species) {
  return SPECIES_EMOJI[String(species || '').toLowerCase()] || '⚠️';
}

/**
 * Build the alarm card's contents from a backend alert.
 *
 * An alert carries the boxes that caused it under `detections`; confidence and
 * the saved frame come from the first of those. A hand-raised alert (the panic
 * button, source `mobile-panic-button`) has none, so both stay null and the
 * card renders a dash.
 *
 * `imagePath` is left as the backend reported it — an absolute Windows path.
 * Turning it into a URL is predatorImageUrl's job, and lives in the service
 * layer so this module stays pure.
 */
export function alarmFromAlert(alert) {
  if (!alert) return null;
  const first = (alert.detections && alert.detections[0]) || null;
  return {
    key: alert.id,
    species: alert.species,
    confidence: first ? first.confidence : null,
    similarity: first ? first.similarity : null,
    imagePath: first ? first.image_path : null,
    imageUri: null,
    at: alert.timestamp,
    source: alert.source || 'camera',
    severity: alert.severity || 'high',
    message: alert.message || `${speciesLabel(alert.species)} detected on camera.`,
  };
}

/**
 * Build the alarm card's contents from a /api/detect response.
 *
 * The scan endpoint returns every accepted box; the alarm reports the most
 * confident one. `previewUri` is the photo the user just picked — it is always
 * displayable, unlike `image_path`, which the detector leaves null whenever its
 * own save cooldown skipped writing the frame.
 *
 * @returns the alarm details, or null when the scan found nothing dangerous.
 */
export function alarmFromScan(result, previewUri = null) {
  if (!result || !result.found) return null;

  const hits = (result.detections || []).filter(d => isDangerSpecies(d.species));
  if (!hits.length) return null;

  const best = hits.reduce((a, b) => ((b.confidence || 0) > (a.confidence || 0) ? b : a));
  return {
    key: `scan_${best.species}_${Date.now()}`,
    species: best.species,
    confidence: best.confidence,
    similarity: best.similarity,
    imagePath: best.image_path || null,
    imageUri: previewUri,
    at: new Date().toISOString(),
    source: 'photo-scan',
    severity: 'high',
    message: `${speciesLabel(best.species)} found in the scanned photo.`,
  };
}

/**
 * Fold one /api/status response's `latest_alert` into the alarm state.
 *
 * @param {{seenAlertId: ?string, armed: boolean}} prev  state after the last poll
 * @param {?object} alert  status.latest_alert, straight from the backend
 * @returns {{seenAlertId: ?string, armed: boolean, fire: boolean, alarm: ?object}}
 *   `fire` is true on the single poll that should start the alarm, and `alarm`
 *   then carries what the card shows.
 *
 * Two things this has to get right:
 *
 * 1. The first response is a BASELINE, never a trigger. /api/status hands back
 *    the newest alert in the backend's history whether it fired ten seconds or
 *    ten hours ago, so without this, opening the screen would scream about an
 *    alert from this morning.
 * 2. After that, an alert id is acted on exactly once. Status is polled every
 *    3s and keeps returning the same `latest_alert` for as long as it is the
 *    newest, so the id — not the species — is what stops the siren restarting
 *    on every poll.
 */
export function nextPredatorAlarmState(prev, alert) {
  const state = prev || IDLE_PREDATOR_ALARM_STATE;
  const id = alert ? alert.id : null;

  if (!state.armed) {
    return { seenAlertId: id, armed: true, fire: false, alarm: null };
  }

  if (!id || id === state.seenAlertId) {
    return { seenAlertId: state.seenAlertId, armed: true, fire: false, alarm: null };
  }

  // A new id, so a new event. Record it either way — a non-danger species must
  // not be re-examined on the next poll.
  const fire = isDangerSpecies(alert.species);
  return {
    seenAlertId: id,
    armed: true,
    fire,
    alarm: fire ? alarmFromAlert(alert) : null,
  };
}
