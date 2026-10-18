// The abnormal-alarm state machine for the chicken behaviour module.
//
// Kept pure and free of React so the rule can be reasoned about (and tested) on
// its own: the alarm must ignore a single abnormal check and only sound once
// two land back to back.

/** ABNORMAL checks in a row required before the alarm sounds. */
export const ALARM_STREAK = 2;

/** Results that end an abnormal run and re-arm the alarm. */
const RESET_RESULTS = ['NORMAL', 'UNDETECTED'];

/** The state an unmonitored / freshly mounted screen starts from. */
export const IDLE_ALARM_STATE = { streak: 0, latched: false };

/**
 * Fold one finished AI check into the alarm state.
 *
 * @param {{streak: number, latched: boolean}} prev  state after the last check
 * @param {string} result  the check's `result` field, straight from /api/status
 * @returns {{streak: number, latched: boolean, fire: boolean, reset: boolean}}
 *   `fire` is true on the single transition that should start the alarm;
 *   `reset` is true when behaviour returned to normal and the alarm re-arms.
 *
 * `latched` is what stops a long abnormal stretch from re-triggering the alarm
 * on every subsequent check — it stays set until a NORMAL/UNDETECTED result
 * clears it.
 */
export function nextAlarmState(prev, result) {
  const state = prev || IDLE_ALARM_STATE;

  if (result === 'ABNORMAL') {
    const streak = state.streak + 1;
    const reached = streak >= ALARM_STREAK;
    return {
      streak,
      latched: state.latched || reached,
      fire: reached && !state.latched,
      reset: false,
    };
  }

  if (RESET_RESULTS.includes(result)) {
    return { streak: 0, latched: false, fire: false, reset: true };
  }

  // Anything else (WAITING, an unknown value, a check that errored) leaves the
  // run untouched rather than silently breaking the streak.
  return { streak: state.streak, latched: state.latched, fire: false, reset: false };
}
