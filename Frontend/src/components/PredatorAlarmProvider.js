import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import PredatorAlarm from './PredatorAlarm';
import {
  fetchPredatorStatus,
  fetchPredatorCaptures,
  predatorImageUrl,
} from '../services/predatorApi';
import {
  IDLE_PREDATOR_ALARM_STATE,
  nextPredatorAlarmState,
} from '../utils/predatorAlarmState';

// The danger alarm has to outlive the Predator Detection screen.
//
// It used to live inside that screen, which is a pushed route - so it was
// unmounted the moment the farmer navigated anywhere else, and a snake found
// while they were on the Dashboard sounded nothing at all. Worse, arriving at
// the screen afterwards adopted that alert as its baseline (see
// nextPredatorAlarmState) and swallowed it for good.
//
// Mounting this provider above the navigator fixes both: one watcher, alive for
// the whole session, so any alert raised after app start sounds on whatever
// screen the farmer happens to be looking at. RN's Modal renders above the rest
// of the app, so the card covers whatever is on top.
//
// This mirrors how the chicken module gets its always-on alarm (the copy
// embedded in the permanently-mounted DashboardScreen owns it), without
// touching that module.

/** How often the whole app checks for a new predator alert. */
const STATUS_POLL_MS = 3000;

// /api/captures/{species} only accepts these two - it 404s on `both`, which is
// what the panic button raises.
const CAPTURE_SPECIES = ['snake', 'scorpion'];

const PredatorAlarmContext = createContext({
  raiseAlarm: () => {},
  stopAlarm: () => {},
  toggleMute: () => {},
  muted: false,
});

/** Screens use this to raise an alarm the poll cannot see (e.g. a photo scan). */
export function usePredatorAlarm() {
  return useContext(PredatorAlarmContext);
}

export default function PredatorAlarmProvider({ children }) {
  const [alarm, setAlarm] = useState(null);
  const [alarmImage, setAlarmImage] = useState(null);
  const [alarmImageLoading, setAlarmImageLoading] = useState(false);
  const [muted, setMuted] = useState(false);

  const alarmStateRef = useRef(IDLE_PREDATOR_ALARM_STATE);
  const alarmKeyRef = useRef(null);

  // Raising the alarm also settles which picture it shows. A photo scan always
  // has one (the image the user just picked). A camera alert only carries a
  // servable frame when the detector's own save cooldown let it write one; when
  // it did not, the newest capture for that species is the next best thing, and
  // the existing /api/captures/{species} serves exactly that.
  const raiseAlarm = useCallback(async details => {
    if (!details) return;
    alarmKeyRef.current = details.key;
    setAlarm(details);
    setAlarmImageLoading(false);

    const direct = details.imageUri || predatorImageUrl(details.imagePath);
    setAlarmImage(direct);
    if (direct || !CAPTURE_SPECIES.includes(details.species)) return;

    setAlarmImageLoading(true);
    try {
      const data = await fetchPredatorCaptures(details.species, 1);
      const file = data && data.files && data.files[0];
      // Ignore a slow response for an alarm the user has already stopped.
      if (file && alarmKeyRef.current === details.key) {
        setAlarmImage(predatorImageUrl(file.url));
      }
    } catch (error) {
      // No frame to show - the card falls back to its species placeholder and
      // the alarm still sounds.
    } finally {
      if (alarmKeyRef.current === details.key) setAlarmImageLoading(false);
    }
  }, []);

  const stopAlarm = useCallback(() => {
    alarmKeyRef.current = null;
    setAlarm(null);
    setAlarmImage(null);
    setAlarmImageLoading(false);
  }, []);

  // Mute is a session preference, not a per-alarm one: it survives the next
  // detection, so silencing the siren once does not have to be done again.
  const toggleMute = useCallback(() => setMuted(m => !m), []);

  // The watch itself. One confirmed snake or scorpion is enough, so this fires
  // straight off /api/status's `latest_alert` with no streak to build up first.
  // A failed poll is ignored on purpose: the Predator screen is what reports a
  // backend outage, and the alarm must not nag from every other screen.
  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      let status;
      try {
        status = await fetchPredatorStatus();
      } catch (error) {
        return;
      }
      if (cancelled) return;

      const next = nextPredatorAlarmState(alarmStateRef.current, status.latest_alert);
      alarmStateRef.current = { seenAlertId: next.seenAlertId, armed: next.armed };
      if (next.fire) raiseAlarm(next.alarm);
    };

    check();
    const timer = setInterval(check, STATUS_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [raiseAlarm]);

  const value = useMemo(
    () => ({ raiseAlarm, stopAlarm, toggleMute, muted }),
    [raiseAlarm, stopAlarm, toggleMute, muted],
  );

  return (
    <PredatorAlarmContext.Provider value={value}>
      {children}
      <PredatorAlarm
        alarm={alarm}
        imageUri={alarmImage}
        imageLoading={alarmImageLoading}
        muted={muted}
        onStop={stopAlarm}
        onToggleMute={toggleMute}
      />
    </PredatorAlarmContext.Provider>
  );
}
