import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
  ScrollView,
  SafeAreaView,
  ActivityIndicator,
  Image,
  RefreshControl,
  Modal,
} from 'react-native';
import { WebView } from 'react-native-webview';
import { LineChart } from 'react-native-chart-kit';
import {
  Camera,
  Activity,
  Clock,
  ChevronLeft,
  Wifi,
  WifiOff,
  Bird,
  RefreshCw,
  TriangleAlert,
  CircleCheck,
  CircleAlert,
  ImageOff,
  Siren,
  VolumeX,
  Volume2,
} from 'lucide-react-native';
import { COLORS, GLASS_STYLE, SHADOW, SHADOW_SM, TYPOGRAPHY, RADIUS } from '../../theme';
import {
  fetchChickenStatus,
  fetchChickenEvents,
  chickenStreamUrl,
  fetchAbnormalHistory,
  abnormalImageUrl,
} from '../services/chickenApi';
import { CHICKEN_API_BASE_URL } from '../config/api';
import { ALARM_STREAK, IDLE_ALARM_STATE, nextAlarmState } from '../utils/alarmState';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
// Card is inset 16 each side and padded 14 each side; leave a little slack so
// the chart can never push the page into horizontal scroll on narrow phones.
const CHART_WIDTH = SCREEN_WIDTH - 64;

const STATUS_POLL_MS = 3000;
const EVENTS_POLL_MS = 6000;
const STREAM_RETRY_MS = 5000;
const EVENTS_LIMIT = 8;
// Saved abnormal frames. Polled far slower than status: a new file can only
// appear once per AI check, and each one is a full-size JPEG.
const ABNORMAL_POLL_MS = 20000;
const ABNORMAL_LIMIT = 1;


const CHART_CONFIG_GREEN = {
  backgroundColor: 'transparent',
  backgroundGradientFrom: '#ffffff',
  backgroundGradientTo: '#f1f8e9',
  decimalPlaces: 0,
  color: (opacity = 1) => `rgba(27, 94, 32, ${opacity})`,
  labelColor: () => COLORS.TEXT_SECONDARY,
  propsForDots: { r: '4', strokeWidth: '2', stroke: COLORS.PRIMARY },
};

const SEVERITY_COLOR = { high: COLORS.DANGER, medium: COLORS.WARNING, low: COLORS.PRIMARY_LIGHT };
const RESULT_TEXT_COLOR = {
  ABNORMAL: COLORS.DANGER,
  NORMAL: COLORS.PRIMARY,
  UNDETECTED: COLORS.WARNING,
  WAITING: COLORS.TEXT_SECONDARY,
};

// Presentation-only mapping: how each backend `result` value is shown.
// The values themselves come from /api/status and are not altered.
const RESULT_META = {
  NORMAL: { label: 'Normal Behaviour', tint: '#E8F5E9', Icon: CircleCheck },
  ABNORMAL: { label: 'Abnormal Behaviour', tint: '#FDECEA', Icon: TriangleAlert },
  UNDETECTED: { label: 'No Chickens Detected', tint: '#FFF3E0', Icon: CircleAlert },
  WAITING: { label: 'Waiting for First Check', tint: '#ECEFF1', Icon: Clock },
};

function metaFor(result) {
  return RESULT_META[result] || RESULT_META.WAITING;
}

function severityFor(result) {
  if (result === 'ABNORMAL') return 'high';
  if (result === 'UNDETECTED') return 'medium';
  return 'low';
}

function formatEventTime(iso) {
  if (!iso) return '-';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const isToday = date.toDateString() === new Date().toDateString();
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  return isToday ? `Today, ${time}` : `${date.toLocaleDateString()}, ${time}`;
}

function formatClock(iso) {
  if (!iso) return '--:--';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '--:--';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** "31 Aug 2026 · 14:32" — the capture stamp on a saved-frame card. */
function formatCaptureTime(iso) {
  if (!iso) return 'Unknown time';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  const day = date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
  return `${day} · ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
}

function streamHtml(url) {
  return `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html,body{margin:0;padding:0;background:#101a10;height:100%;overflow:hidden}
  .wrap{width:100%;height:100%;display:flex;align-items:center;justify-content:center}
  img{max-width:100%;max-height:100%;object-fit:contain}
</style></head>
<body>
  <div class="wrap">
    <img src="${url}"
      onerror="window.ReactNativeWebView && window.ReactNativeWebView.postMessage('error')"
      onload="window.ReactNativeWebView && window.ReactNativeWebView.postMessage('ok')" />
  </div>
</body></html>`;
}

// The alarm tone is generated with the Web Audio API inside a 1px WebView.
// react-native-webview is already a dependency here (it carries the MJPEG
// stream), so the alarm needs no audio library and no native rebuild — it ships
// with a JS reload. startAlarm()/stopAlarm() are driven from RN by
// injectJavaScript. mediaPlaybackRequiresUserAction={false} is what lets it
// sound without a tap.
const ALARM_HTML = `<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;background:transparent">
<script>
  var ctx = null, timer = null;

  function audio() {
    if (!ctx) { ctx = new (window.AudioContext || window.webkitAudioContext)(); }
    if (ctx.state === 'suspended') { ctx.resume(); }
    return ctx;
  }

  // One shaped beep. Ramped rather than switched, so it does not click.
  function beep(freq, at, dur) {
    var c = audio();
    var osc = c.createOscillator();
    var gain = c.createGain();
    var t0 = c.currentTime + at;
    osc.type = 'square';
    osc.frequency.setValueAtTime(freq, t0);
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(0.3, t0 + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(gain);
    gain.connect(c.destination);
    osc.start(t0);
    osc.stop(t0 + dur + 0.03);
  }

  // Two-tone emergency pattern, repeated until stopped.
  function cycle() { beep(880, 0, 0.26); beep(660, 0.3, 0.26); }

  function startAlarm() {
    if (timer) { return; }
    cycle();
    timer = setInterval(cycle, 900);
  }

  function stopAlarm() {
    if (timer) { clearInterval(timer); timer = null; }
  }
</script></body></html>`;

/** Small labelled tile used by the analytics row. */
function StatTile({ icon, label, value }) {
  return (
    <View style={[styles.statTile, GLASS_STYLE, SHADOW_SM]}>
      <View style={styles.statTileHead}>
        {icon}
        <Text style={styles.statTileLabel} numberOfLines={1}>
          {label}
        </Text>
      </View>
      <Text style={styles.statTileValue} numberOfLines={1}>
        {value}
      </Text>
    </View>
  );
}

/**
 * The most recent saved abnormal frame — the same shape the predator screen
 * uses for its last alert snapshot: one card, one image, one caption. The file
 * can go missing (folder cleared, backend moved), so the image keeps its own
 * loading and failure state rather than leaving an empty box.
 */
function AbnormalSnapshotCard({ shot }) {
  const [imageLoading, setImageLoading] = useState(true);
  const [imageFailed, setImageFailed] = useState(false);

  const uri = abnormalImageUrl(shot.filename);
  const confidence =
    typeof shot.confidence === 'number' ? `${(shot.confidence * 100).toFixed(1)}% confidence` : null;
  const caption = [formatCaptureTime(shot.timestamp || shot.captured_at), confidence]
    .filter(Boolean)
    .join('  ·  ');

  return (
    <View style={[styles.captureCard, SHADOW_SM]}>
      <View style={styles.captureImageWrap}>
        {uri && !imageFailed ? (
          <Image
            source={{ uri }}
            style={styles.captureImage}
            resizeMode="contain"
            onLoadEnd={() => setImageLoading(false)}
            onError={() => {
              setImageLoading(false);
              setImageFailed(true);
            }}
          />
        ) : null}

        {(imageFailed || !uri || imageLoading) && (
          <View style={styles.captureImageState}>
            {imageFailed || !uri ? (
              <>
                <ImageOff size={22} color={COLORS.TEXT_SECONDARY} strokeWidth={1.8} />
                <Text style={styles.captureImageStateText}>Image unavailable</Text>
              </>
            ) : (
              <>
                <ActivityIndicator size="small" color={COLORS.PRIMARY} />
                <Text style={styles.captureImageStateText}>Loading snapshot…</Text>
              </>
            )}
          </View>
        )}
      </View>

      <Text style={styles.captureCaption}>{caption}</Text>
    </View>
  );
}

export default function ChickenBehaviourModule({ navigation }) {
  const alertAnim = useRef(new Animated.Value(1)).current;
  const [alertDismissed, setAlertDismissed] = useState(false);
  // Raw = the live camera stream (~6 fps, every frame new). The annotated
  // stream is NOT live: the backend only calls set_annotated() at the end of
  // each 30s AI check, so it re-sends one still until the next check lands.
  // Default to raw so the screen opens on genuine live monitoring.
  const [annotatedView, setAnnotatedView] = useState(false);
  const [streamNonce, setStreamNonce] = useState(0);
  const [streamError, setStreamError] = useState(false);
  const [streamReady, setStreamReady] = useState(false);

  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const [events, setEvents] = useState([]);

  const [abnormalShots, setAbnormalShots] = useState([]);
  const [abnormalLoading, setAbnormalLoading] = useState(true);
  const [abnormalError, setAbnormalError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // ── two-consecutive-abnormal alarm ───────────────────────────────────────
  // Only one mounted copy of this component may own the alarm, otherwise the
  // dashboard's copy and the standalone screen's copy would both sound it. The
  // embedded copy owns it: DashboardScreen stays mounted for the whole session,
  // so the alarm keeps watching while the user is on any screen, and RN's Modal
  // renders above whatever is on top.
  const ownsAlarm = !navigation;
  const [abnormalStreak, setAbnormalStreak] = useState(0);
  const [alarmActive, setAlarmActive] = useState(false);
  const [alarmMuted, setAlarmMuted] = useState(false);
  const [alarmDetails, setAlarmDetails] = useState(null);
  const alarmStateRef = useRef(IDLE_ALARM_STATE);
  const countedCheckRef = useRef(null);
  const alarmWebRef = useRef(null);
  const alarmWebReadyRef = useRef(false);

  const lastAbnormalCheckRef = useRef(null);
  const retryTimerRef = useRef(null);

  const loadStatus = useCallback(async () => {
    try {
      const data = await fetchChickenStatus();
      setStatus(data);
      setStatusError(null);
      if (data.result === 'ABNORMAL' && data.check_number !== lastAbnormalCheckRef.current) {
        lastAbnormalCheckRef.current = data.check_number;
        setAlertDismissed(false);
      }
    } catch (error) {
      setStatusError(error.message || 'Backend unreachable');
    }
  }, []);

  const loadEvents = useCallback(async () => {
    try {
      const data = await fetchChickenEvents(EVENTS_LIMIT);
      setEvents(data.events || []);
    } catch (error) {
      // status polling already surfaces connectivity problems
    }
  }, []);

  const loadAbnormal = useCallback(async () => {
    try {
      const data = await fetchAbnormalHistory(ABNORMAL_LIMIT);
      setAbnormalShots(data.events || []);
      setAbnormalError(null);
    } catch (error) {
      setAbnormalError(error.message || 'Could not load saved captures');
    } finally {
      setAbnormalLoading(false);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    setRefreshing(true);
    await Promise.allSettled([loadStatus(), loadEvents(), loadAbnormal()]);
    setRefreshing(false);
  }, [loadStatus, loadEvents, loadAbnormal]);

  useEffect(() => {
    loadStatus();
    loadEvents();
    loadAbnormal();
    const statusTimer = setInterval(loadStatus, STATUS_POLL_MS);
    const eventsTimer = setInterval(loadEvents, EVENTS_POLL_MS);
    const abnormalTimer = setInterval(loadAbnormal, ABNORMAL_POLL_MS);
    return () => {
      clearInterval(statusTimer);
      clearInterval(eventsTimer);
      clearInterval(abnormalTimer);
    };
  }, [loadStatus, loadEvents, loadAbnormal]);

  // Counts CHECKS, not polls. /api/status is read every 3s but a new AI check
  // only lands every ~30s, so each check_number is counted exactly once. The
  // comparison is inequality, not >, so a backend restart (check_number back to
  // 1) still counts instead of being silently ignored.
  useEffect(() => {
    if (!ownsAlarm || !status) return;

    const check = status.check_number;
    if (check == null || check === countedCheckRef.current) return;
    countedCheckRef.current = check;

    const next = nextAlarmState(alarmStateRef.current, status.result);
    alarmStateRef.current = { streak: next.streak, latched: next.latched };
    setAbnormalStreak(next.streak);

    if (next.fire) {
      setAlarmDetails({
        checkNumber: check,
        confidence: status.confidence,
        at: status.last_check_at,
        classifier: status.classifier_details,
        streak: next.streak,
      });
      setAlarmMuted(false);
      setAlarmActive(true);
      loadAbnormal(); // pull the frame this alarm is about
    }

    if (next.reset) {
      setAlarmActive(false);
      setAlarmMuted(false);
    }
  }, [status, ownsAlarm, loadAbnormal]);

  const alarmSounding = ownsAlarm && alarmActive && !alarmMuted;

  useEffect(() => {
    if (!ownsAlarm || !alarmWebReadyRef.current) return;
    alarmWebRef.current?.injectJavaScript(
      `${alarmSounding ? 'startAlarm' : 'stopAlarm'}(); true;`
    );
  }, [alarmSounding, ownsAlarm]);

  useEffect(
    () => () => {
      alarmWebRef.current?.injectJavaScript('stopAlarm(); true;');
    },
    []
  );

  const silenceAlarm = useCallback(() => setAlarmMuted(true), []);
  const dismissAlarm = useCallback(() => {
    setAlarmActive(false);
    setAlarmMuted(false);
  }, []);

  useEffect(() => {
    const isAbnormalNow = status?.result === 'ABNORMAL';
    if (!isAbnormalNow || alertDismissed) {
      alertAnim.stopAnimation();
      alertAnim.setValue(1);
      return undefined;
    }
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(alertAnim, { toValue: 0.15, duration: 480, useNativeDriver: true }),
        Animated.timing(alertAnim, { toValue: 1, duration: 480, useNativeDriver: true }),
      ])
    );
    anim.start();
    return () => anim.stop();
  }, [status?.result, alertDismissed, alertAnim]);

  useEffect(() => {
    return () => {
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    };
  }, []);

  const handleStreamMessage = useCallback((event) => {
    const message = event.nativeEvent.data;
    if (message === 'error') {
      setStreamError(true);
      setStreamReady(false);
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      retryTimerRef.current = setTimeout(() => {
        setStreamNonce((n) => n + 1);
      }, STREAM_RETRY_MS);
    } else {
      setStreamError(false);
      setStreamReady(true);
    }
  }, []);

  const retryStreamNow = useCallback(() => {
    setStreamError(false);
    setStreamReady(false);
    setStreamNonce((n) => n + 1);
  }, []);

  const toggleAnnotated = useCallback(() => {
    setStreamReady(false);
    setAnnotatedView((v) => !v);
  }, []);

  const baseStreamUrl = chickenStreamUrl(annotatedView);
  const separator = baseStreamUrl.includes('?') ? '&' : '?';
  const streamSrc = `${baseStreamUrl}${separator}_r=${streamNonce}`;

  const isAbnormal = status?.result === 'ABNORMAL';
  const cameraConnected = !!status?.camera_connected && !statusError;
  const isLoading = !status && !statusError;

  const result = status?.result || 'WAITING';
  const meta = metaFor(result);
  const resultColor = RESULT_TEXT_COLOR[result] || COLORS.TEXT_PRIMARY;
  const MetaIcon = meta.Icon;
  const confidencePct = status ? Math.round((status.confidence || 0) * 100) : 0;

  const latestShot = abnormalShots[0] || null;

  const chartEvents = [...events].reverse().slice(-EVENTS_LIMIT);
  const hasChartData = chartEvents.length > 0;
  const chartData = {
    labels: hasChartData ? chartEvents.map((e) => formatClock(e.timestamp)) : ['--'],
    datasets: [
      {
        data: hasChartData ? chartEvents.map((e) => Math.round((e.confidence || 0) * 100)) : [0],
        color: (opacity = 1) => `rgba(27, 94, 32, ${opacity})`,
        strokeWidth: 2,
      },
    ],
  };

  const body = (
    <>
      {/* ─── Live monitoring ─────────────────────────────────────────────── */}
      <View style={styles.blockHeader}>
        <View style={styles.blockTitleRow}>
          <View style={styles.sectionDot} />
          <Text style={styles.blockTitle}>Live Monitoring</Text>
        </View>
        <View
          style={[
            styles.camPill,
            { backgroundColor: cameraConnected ? '#E8F5E9' : '#FDECEA' },
          ]}
        >
          {cameraConnected ? (
            <Wifi size={13} color={COLORS.PRIMARY} strokeWidth={2.2} />
          ) : (
            <WifiOff size={13} color={COLORS.DANGER} strokeWidth={2.2} />
          )}
          <Text
            style={[styles.camPillText, { color: cameraConnected ? COLORS.PRIMARY : COLORS.DANGER }]}
          >
            {cameraConnected ? 'Camera Connected' : 'Camera Offline'}
          </Text>
        </View>
      </View>

      <View style={styles.cameraCard}>
        {isAbnormal && !alertDismissed && (
          <Animated.View style={[styles.alertBanner, { opacity: alertAnim }]}>
            <TriangleAlert size={15} color={COLORS.WHITE} strokeWidth={2.4} />
            <Text style={styles.alertBannerText}>ABNORMAL BEHAVIOUR DETECTED</Text>
          </Animated.View>
        )}

        <View style={styles.cameraFeed}>
          <WebView
            key={streamSrc}
            // baseUrl gives the document the backend's own origin; without it
            // Android loads the HTML with an opaque origin and the MJPEG <img>
            // never paints.
            source={{ html: streamHtml(streamSrc), baseUrl: `${CHICKEN_API_BASE_URL}/` }}
            originWhitelist={['*']}
            mixedContentMode="always"
            style={styles.webview}
            scrollEnabled={false}
            onMessage={handleStreamMessage}
          />

          {/* Connecting state — sits over the feed until the first frame paints */}
          {!streamReady && !streamError && (
            <View style={styles.streamOverlay} pointerEvents="none">
              <ActivityIndicator size="small" color={COLORS.WHITE} />
              <Text style={styles.streamOverlayText}>Connecting to camera…</Text>
            </View>
          )}

          {/* Live badge */}
          <View style={styles.liveBadge}>
            <View
              style={[
                styles.liveDot,
                { backgroundColor: cameraConnected ? COLORS.PRIMARY_LIGHT : COLORS.DANGER },
              ]}
            />
            <Text style={styles.liveText}>
              {!cameraConnected ? 'OFFLINE' : annotatedView ? 'AI VIEW' : 'LIVE'}
            </Text>
          </View>

          {streamError && (
            <TouchableOpacity
              style={styles.streamErrorOverlay}
              onPress={retryStreamNow}
              activeOpacity={0.85}
            >
              <CircleAlert size={26} color={COLORS.WHITE} strokeWidth={2} />
              <Text style={styles.streamErrorText}>Camera stream unreachable</Text>
              <View style={styles.retryChip}>
                <RefreshCw size={13} color={COLORS.WHITE} strokeWidth={2.2} />
                <Text style={styles.retryChipText}>Tap to retry</Text>
              </View>
            </TouchableOpacity>
          )}
        </View>
      </View>

      {/* ─── Loading / error / data ──────────────────────────────────────── */}
      {isLoading ? (
        <View style={[styles.stateCard, GLASS_STYLE, SHADOW_SM]}>
          <ActivityIndicator size="small" color={COLORS.PRIMARY} />
          <Text style={styles.stateText}>Loading chicken behaviour data...</Text>
        </View>
      ) : statusError ? (
        <View style={[styles.errorCard, SHADOW_SM]}>
          <View style={styles.errorHead}>
            <TriangleAlert size={17} color={COLORS.DANGER} strokeWidth={2.2} />
            <Text style={styles.errorTitle}>Unable to connect to chicken monitoring service</Text>
          </View>
          <Text style={styles.errorBody}>
            {statusError}. Check CHICKEN_API_BASE_URL in src/config/api.js and make sure this device
            is on the same network as the detector.
          </Text>
        </View>
      ) : (
        <>
          {/* Detection status */}
          <View style={[styles.statusCard, GLASS_STYLE, SHADOW]}>
            <Text style={styles.cardTitle}>Chicken Behaviour Status</Text>

            <View style={styles.statusRow}>
              <View style={[styles.statusIconWrap, { backgroundColor: meta.tint }]}>
                <MetaIcon size={22} color={resultColor} strokeWidth={2.2} />
              </View>
              <View style={styles.statusTextWrap}>
                <Text style={[styles.statusLabel, { color: resultColor }]} numberOfLines={2}>
                  {meta.label}
                </Text>
              </View>
            </View>

            <View style={styles.confHeader}>
              <Text style={styles.confLabel}>Confidence</Text>
              <Text style={[styles.confValue, { color: resultColor }]}>{confidencePct}%</Text>
            </View>
            <View style={styles.confTrack}>
              <View
                style={[styles.confFill, { width: `${confidencePct}%`, backgroundColor: resultColor }]}
              />
            </View>

            <View style={styles.statusFooter}>
              <Clock size={13} color={COLORS.TEXT_SECONDARY} strokeWidth={2} />
              <Text style={styles.statusFooterText}>
                Last updated {formatClock(status?.last_check_at)}
              </Text>
            </View>
          </View>

          {/* Analytics */}
          <Text style={styles.subHeading}>Monitoring Analytics</Text>
          <View style={styles.statRow}>
            <StatTile
              icon={<Activity size={14} color={COLORS.PRIMARY} strokeWidth={2.2} />}
              label="Check"
              value={`#${status?.check_number ?? 0}`}
            />
            <StatTile
              icon={<Clock size={14} color={COLORS.WARNING} strokeWidth={2.2} />}
              label="Next check"
              value={status?.next_check_in != null ? `${status.next_check_in}s` : '—'}
            />
          </View>

          {/* Confidence trend */}
          <View style={[styles.chartCard, GLASS_STYLE, SHADOW_SM]}>
            <Text style={styles.cardTitle}>Confidence Trend</Text>
            {hasChartData ? (
              <LineChart
                data={chartData}
                width={CHART_WIDTH}
                height={170}
                chartConfig={CHART_CONFIG_GREEN}
                bezier
                style={styles.lineChart}
                withVerticalLines={false}
                withHorizontalLines
                fromZero
                yAxisSuffix="%"
              />
            ) : (
              <View style={styles.emptyBox}>
                <Activity size={20} color={COLORS.TEXT_SECONDARY} strokeWidth={1.8} />
                <Text style={styles.emptyText}>No detection checks yet — waiting for backend…</Text>
              </View>
            )}
          </View>
        </>
      )}

      {/* ─── Actions ─────────────────────────────────────────────────────── */}
      <View style={styles.actionsGrid}>
        <TouchableOpacity
          style={[styles.actionBtn, { backgroundColor: COLORS.PRIMARY }]}
          onPress={toggleAnnotated}
          activeOpacity={0.85}
        >
          <Camera size={15} color={COLORS.WHITE} strokeWidth={2.2} />
          <Text style={styles.actionBtnText} numberOfLines={1}>
            {annotatedView ? 'AI Boxes · per check' : 'Live View'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.actionBtn,
            { backgroundColor: alertDismissed || !isAbnormal ? '#9E9E9E' : COLORS.DANGER },
          ]}
          onPress={() => setAlertDismissed(true)}
          activeOpacity={0.85}
          disabled={!isAbnormal || alertDismissed}
        >
          {isAbnormal && !alertDismissed ? (
            <TriangleAlert size={15} color={COLORS.WHITE} strokeWidth={2.2} />
          ) : (
            <CircleCheck size={15} color={COLORS.WHITE} strokeWidth={2.2} />
          )}
          <Text style={styles.actionBtnText} numberOfLines={1}>
            {isAbnormal && !alertDismissed ? 'Dismiss Alert' : 'No Active Alert'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* ─── Event history ───────────────────────────────────────────────── */}
      <View style={styles.timelineSection}>
        <Text style={styles.subHeadingFlush}>Event History</Text>

        {isLoading && (
          <View style={[styles.stateCardFlush, GLASS_STYLE, SHADOW_SM]}>
            <ActivityIndicator size="small" color={COLORS.PRIMARY} />
            <Text style={styles.stateText}>Loading chicken behaviour data...</Text>
          </View>
        )}

        {!isLoading && events.length === 0 && (
          <View style={[styles.stateCardFlush, GLASS_STYLE, SHADOW_SM]}>
            <Bird size={20} color={COLORS.TEXT_SECONDARY} strokeWidth={1.8} />
            <Text style={styles.stateText}>No behaviour events detected</Text>
          </View>
        )}

        {events.slice(0, 6).map((event) => {
          const severity = severityFor(event.result);
          const tone = SEVERITY_COLOR[severity];
          const eventMeta = metaFor(event.result);
          return (
            <View key={event.check_number} style={[styles.alertCard, SHADOW_SM]}>
              <View style={[styles.severityBar, { backgroundColor: tone }]} />
              <View style={styles.alertContent}>
                <View style={styles.alertTopRow}>
                  <Text style={styles.alertMsg} numberOfLines={1}>
                    {eventMeta.label}
                  </Text>
                  <View style={[styles.resolvedBadge, { backgroundColor: `${tone}22` }]}>
                    <Text style={[styles.resolvedText, { color: tone }]}>{event.result}</Text>
                  </View>
                </View>
                <View style={styles.alertMeta}>
                  <Text style={styles.alertTime} numberOfLines={1}>
                    {formatEventTime(event.timestamp)}
                  </Text>
                  <Text style={styles.alertConf}>
                    {Math.round((event.confidence || 0) * 100)}%
                  </Text>
                </View>
              </View>
            </View>
          );
        })}
      </View>

      {/* ─── Alarm: sound + popup, owned by one copy ─────────────────────── */}
      {ownsAlarm && (
        <View style={styles.alarmWebWrap} pointerEvents="none">
          <WebView
            ref={alarmWebRef}
            source={{ html: ALARM_HTML }}
            originWhitelist={['*']}
            javaScriptEnabled
            mediaPlaybackRequiresUserAction={false}
            allowsInlineMediaPlayback
            style={styles.alarmWeb}
            onLoadEnd={() => {
              alarmWebReadyRef.current = true;
              if (alarmSounding) {
                alarmWebRef.current?.injectJavaScript('startAlarm(); true;');
              }
            }}
          />
        </View>
      )}

      {ownsAlarm && (
        <Modal
          visible={alarmActive}
          transparent
          animationType="fade"
          onRequestClose={dismissAlarm}
        >
          <View style={styles.alarmBackdrop}>
            <View style={[styles.alarmCard, SHADOW]}>
              <View style={styles.alarmHeader}>
                <Siren size={20} color={COLORS.WHITE} strokeWidth={2.2} />
                <Text style={styles.alarmHeaderText} numberOfLines={1}>
                  Abnormal Behaviour Alarm
                </Text>
              </View>

              <Text style={styles.alarmLead}>
                {alarmDetails ? alarmDetails.streak : ALARM_STREAK} consecutive abnormal checks
                detected.
              </Text>

              {latestShot ? (
                <AbnormalSnapshotCard shot={latestShot} />
              ) : (
                <View style={[styles.captureEmpty, GLASS_STYLE]}>
                  <ActivityIndicator size="small" color={COLORS.DANGER} />
                  <Text style={styles.captureEmptyText}>Fetching the captured frame…</Text>
                </View>
              )}

              <View style={styles.alarmFacts}>
                <View style={styles.alarmFact}>
                  <Text style={styles.alarmFactLabel}>CONFIDENCE</Text>
                  <Text style={[styles.alarmFactValue, { color: COLORS.DANGER }]}>
                    {alarmDetails && typeof alarmDetails.confidence === 'number'
                      ? `${(alarmDetails.confidence * 100).toFixed(1)}%`
                      : '—'}
                  </Text>
                </View>
                <View style={styles.alarmFact}>
                  <Text style={styles.alarmFactLabel}>CHECK</Text>
                  <Text style={styles.alarmFactValue}>
                    #{alarmDetails ? alarmDetails.checkNumber : '—'}
                  </Text>
                </View>
              </View>

              {alarmDetails?.at ? (
                <Text style={styles.alarmMeta}>{formatCaptureTime(alarmDetails.at)}</Text>
              ) : null}
              {alarmDetails?.classifier ? (
                <Text style={styles.alarmMeta} numberOfLines={2}>
                  {alarmDetails.classifier}
                </Text>
              ) : null}

              <View style={styles.alarmButtons}>
                <TouchableOpacity
                  style={[
                    styles.alarmBtn,
                    { backgroundColor: alarmMuted ? '#9E9E9E' : COLORS.DANGER },
                  ]}
                  onPress={silenceAlarm}
                  disabled={alarmMuted}
                  activeOpacity={0.85}
                >
                  {alarmMuted ? (
                    <VolumeX size={15} color={COLORS.WHITE} strokeWidth={2.2} />
                  ) : (
                    <Volume2 size={15} color={COLORS.WHITE} strokeWidth={2.2} />
                  )}
                  <Text style={styles.alarmBtnText}>
                    {alarmMuted ? 'Alarm muted' : 'Stop alarm'}
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.alarmBtn, styles.alarmBtnGhost]}
                  onPress={dismissAlarm}
                  activeOpacity={0.85}
                >
                  <Text style={[styles.alarmBtnText, { color: COLORS.TEXT_PRIMARY }]}>
                    Dismiss
                  </Text>
                </TouchableOpacity>
              </View>

              <Text style={styles.alarmFootnote}>
                The alarm re-arms by itself once behaviour returns to NORMAL.
              </Text>
            </View>
          </View>
        </Modal>
      )}

      {/* ─── Last abnormal snapshot ──────────────────────────────────────── */}
      <View style={styles.captureSection}>
        <View style={styles.captureSectionHead}>
          <View style={styles.captureSectionTitleRow}>
            <TriangleAlert size={15} color={COLORS.DANGER} strokeWidth={2.3} />
            <Text style={styles.captureSectionTitle}>Last Abnormal Snapshot</Text>
          </View>
          {ownsAlarm && abnormalStreak > 0 ? (
            <View style={styles.captureCountPill}>
              <Text style={styles.captureCountText}>
                Abnormal streak {Math.min(abnormalStreak, ALARM_STREAK)}/{ALARM_STREAK}
              </Text>
            </View>
          ) : (
            latestShot &&
            latestShot.check_number != null && (
              <View style={styles.captureCountPill}>
                <Text style={styles.captureCountText}>Check #{latestShot.check_number}</Text>
              </View>
            )
          )}
        </View>

        {abnormalLoading && !latestShot && (
          <View style={[styles.stateCardFlush, GLASS_STYLE, SHADOW_SM]}>
            <ActivityIndicator size="small" color={COLORS.DANGER} />
            <Text style={styles.stateText}>Loading snapshot…</Text>
          </View>
        )}

        {!abnormalLoading && abnormalError && !latestShot && (
          <View style={[styles.stateCardFlush, GLASS_STYLE, SHADOW_SM]}>
            <ImageOff size={20} color={COLORS.DANGER} strokeWidth={1.8} />
            <Text style={styles.stateText} numberOfLines={2}>
              {abnormalError}
            </Text>
            <TouchableOpacity
              style={styles.captureRetryBtn}
              onPress={loadAbnormal}
              activeOpacity={0.85}
            >
              <RefreshCw size={13} color={COLORS.WHITE} strokeWidth={2.2} />
              <Text style={styles.captureRetryText}>Retry</Text>
            </TouchableOpacity>
          </View>
        )}

        {!abnormalLoading && !abnormalError && !latestShot && (
          <View style={[styles.captureEmpty, GLASS_STYLE, SHADOW_SM]}>
            <View style={styles.captureEmptyIcon}>
              <CircleCheck size={22} color={COLORS.PRIMARY} strokeWidth={2} />
            </View>
            <Text style={styles.captureEmptyTitle}>No abnormal snapshot yet</Text>
            <Text style={styles.captureEmptyText}>
              Nothing has been flagged. The frame is captured automatically the next time the
              flock is scored ABNORMAL.
            </Text>
          </View>
        )}

        {latestShot && <AbnormalSnapshotCard key={latestShot.filename} shot={latestShot} />}
      </View>
    </>
  );

  // Rendered two ways: as a section inside DashboardScreen's ScrollView (no
  // navigation prop), and as its own stack screen. Only the standalone case
  // supplies page chrome — wrapping unconditionally would nest scroll views
  // inside the dashboard.
  if (navigation) {
    return (
      <SafeAreaView style={styles.page}>
        <View style={styles.pageHeader}>
          <TouchableOpacity
            style={styles.backBtn}
            onPress={() => navigation.goBack()}
            activeOpacity={0.7}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <ChevronLeft size={22} color={COLORS.TEXT_PRIMARY} strokeWidth={2.2} />
          </TouchableOpacity>
          <Text style={styles.pageTitle} numberOfLines={1}>
            Chicken Behaviour
          </Text>
          <View style={styles.backBtn} />
        </View>
        <ScrollView
          style={styles.pageScroll}
          contentContainerStyle={styles.pageScrollContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={refreshAll}
              colors={[COLORS.PRIMARY]}
              tintColor={COLORS.PRIMARY}
            />
          }
        >
          {body}
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <View>
      <View style={styles.sectionHeader}>
        <View style={styles.sectionDot} />
        <Text style={styles.sectionTitle}>Chicken Behaviour Detection</Text>
      </View>
      {body}
    </View>
  );
}

const styles = StyleSheet.create({
  // ── standalone page chrome ────────────────────────────────────────────────
  page: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  pageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: COLORS.WHITE,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0,0,0,0.06)',
  },
  backBtn: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pageTitle: {
    flex: 1,
    textAlign: 'center',
    fontSize: TYPOGRAPHY.LG,
    fontWeight: '700',
    color: COLORS.TEXT_PRIMARY,
  },
  pageScroll: {
    flex: 1,
  },
  pageScrollContent: {
    paddingTop: 6,
    paddingBottom: 32,
  },

  // ── section headings ──────────────────────────────────────────────────────
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 16,
    marginTop: 20,
    marginBottom: 12,
    gap: 8,
  },
  sectionDot: {
    width: 4,
    height: 18,
    borderRadius: 2,
    backgroundColor: COLORS.PRIMARY,
  },
  sectionTitle: {
    fontSize: TYPOGRAPHY.LG,
    fontWeight: '700',
    color: COLORS.TEXT_PRIMARY,
    letterSpacing: 0.2,
  },
  blockHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: 16,
    marginTop: 14,
    marginBottom: 10,
    gap: 10,
  },
  blockTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flexShrink: 1,
  },
  blockTitle: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  camPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 9,
    paddingVertical: 5,
    borderRadius: 999,
  },
  camPillText: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
  },
  subHeading: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginHorizontal: 16,
    marginTop: 20,
    marginBottom: 10,
  },
  subHeadingFlush: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 10,
  },

  // ── live monitoring ───────────────────────────────────────────────────────
  cameraCard: {
    marginHorizontal: 16,
    borderRadius: RADIUS.LG,
    overflow: 'hidden',
    backgroundColor: '#101a10',
    ...SHADOW,
  },
  alertBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    backgroundColor: COLORS.DANGER,
    paddingVertical: 9,
  },
  alertBannerText: {
    color: COLORS.WHITE,
    fontWeight: '800',
    fontSize: TYPOGRAPHY.MD,
    letterSpacing: 0.6,
  },
  cameraFeed: {
    width: '100%',
    aspectRatio: 16 / 9,
    backgroundColor: '#101a10',
    position: 'relative',
  },
  webview: {
    flex: 1,
    backgroundColor: '#101a10',
  },
  liveBadge: {
    position: 'absolute',
    top: 10,
    left: 12,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
    gap: 5,
  },
  liveDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  liveText: {
    color: COLORS.WHITE,
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    letterSpacing: 1,
  },
  streamOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: 'rgba(16,26,16,0.75)',
  },
  streamOverlayText: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '600',
  },
  streamErrorOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.68)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: 20,
  },
  streamErrorText: {
    color: COLORS.WHITE,
    fontWeight: '700',
    textAlign: 'center',
    fontSize: TYPOGRAPHY.MD,
  },
  retryChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.18)',
  },
  retryChipText: {
    color: COLORS.WHITE,
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
  },

  // ── states ────────────────────────────────────────────────────────────────
  stateCard: {
    marginHorizontal: 16,
    marginTop: 14,
    paddingVertical: 22,
    paddingHorizontal: 16,
    alignItems: 'center',
    gap: 10,
  },
  stateCardFlush: {
    paddingVertical: 22,
    paddingHorizontal: 16,
    alignItems: 'center',
    gap: 10,
  },
  stateText: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '600',
    textAlign: 'center',
  },
  errorCard: {
    marginHorizontal: 16,
    marginTop: 14,
    padding: 14,
    borderRadius: RADIUS.MD,
    backgroundColor: '#FDECEA',
    borderWidth: 1,
    borderColor: COLORS.DANGER,
  },
  errorHead: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginBottom: 6,
  },
  errorTitle: {
    flex: 1,
    color: COLORS.DANGER,
    fontWeight: '800',
    fontSize: TYPOGRAPHY.MD,
    lineHeight: 19,
  },
  errorBody: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.SM,
    lineHeight: 18,
  },

  // ── detection status card ─────────────────────────────────────────────────
  statusCard: {
    marginHorizontal: 16,
    marginTop: 14,
    padding: 16,
  },
  cardTitle: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    color: COLORS.TEXT_SECONDARY,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 12,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  statusIconWrap: {
    width: 46,
    height: 46,
    borderRadius: 23,
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusTextWrap: {
    flex: 1,
  },
  statusLabel: {
    fontSize: TYPOGRAPHY.XL,
    fontWeight: '800',
    letterSpacing: 0.2,
  },
  confHeader: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    marginTop: 16,
    marginBottom: 6,
  },
  confLabel: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    color: COLORS.TEXT_SECONDARY,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  confValue: {
    fontSize: TYPOGRAPHY.XL,
    fontWeight: '800',
  },
  confTrack: {
    height: 7,
    borderRadius: 4,
    backgroundColor: 'rgba(0,0,0,0.08)',
    overflow: 'hidden',
  },
  confFill: {
    height: '100%',
    borderRadius: 4,
  },
  statusFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 12,
  },
  statusFooterText: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
  },

  // ── analytics tiles ───────────────────────────────────────────────────────
  statRow: {
    flexDirection: 'row',
    marginHorizontal: 16,
    gap: 10,
  },
  statTile: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 10,
  },
  statTileHead: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginBottom: 6,
  },
  statTileLabel: {
    flexShrink: 1,
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    color: COLORS.TEXT_SECONDARY,
  },
  statTileValue: {
    fontSize: TYPOGRAPHY.XL,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
  },

  // ── chart ─────────────────────────────────────────────────────────────────
  chartCard: {
    marginHorizontal: 16,
    marginTop: 14,
    padding: 14,
  },
  lineChart: {
    borderRadius: RADIUS.MD,
    marginTop: 4,
    marginLeft: -8,
  },
  emptyBox: {
    alignItems: 'center',
    gap: 8,
    paddingVertical: 18,
  },
  emptyText: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.SM,
    textAlign: 'center',
  },

  // ── actions ───────────────────────────────────────────────────────────────
  actionsGrid: {
    flexDirection: 'row',
    marginHorizontal: 16,
    marginTop: 16,
    gap: 10,
  },
  actionBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    minHeight: 48,
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderRadius: RADIUS.MD,
    ...SHADOW_SM,
  },
  actionBtnText: {
    flexShrink: 1,
    color: COLORS.WHITE,
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    textAlign: 'center',
  },

  // ── event history ─────────────────────────────────────────────────────────
  timelineSection: {
    marginHorizontal: 16,
    marginTop: 22,
    marginBottom: 24,
  },
  alertCard: {
    flexDirection: 'row',
    backgroundColor: COLORS.CARD_BG,
    borderRadius: RADIUS.MD,
    marginBottom: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.06)',
  },
  severityBar: {
    width: 4,
  },
  alertContent: {
    flex: 1,
    padding: 12,
  },
  alertTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  alertMsg: {
    flex: 1,
    fontSize: TYPOGRAPHY.MD,
    color: COLORS.TEXT_PRIMARY,
    fontWeight: '700',
  },
  alertMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
    marginTop: 6,
  },
  alertTime: {
    flexShrink: 1,
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
  },
  alertConf: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: '600',
  },
  resolvedBadge: {
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  resolvedText: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
  },

  // ── last abnormal snapshot ────────────────────────────────────────────────
  captureSection: {
    marginHorizontal: 16,
    marginBottom: 28,
  },
  captureSectionHead: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
    marginBottom: 10,
  },
  captureSectionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    flexShrink: 1,
  },
  captureSectionTitle: {
    fontSize: TYPOGRAPHY.LG,
    fontWeight: '700',
    color: COLORS.TEXT_PRIMARY,
  },
  captureCountPill: {
    backgroundColor: '#FDECEA',
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 3,
  },
  captureCountText: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    color: COLORS.DANGER,
  },
  captureCard: {
    backgroundColor: COLORS.WHITE,
    borderRadius: RADIUS.LG,
    padding: 10,
    borderWidth: 1,
    borderColor: 'rgba(211,47,47,0.18)',
  },
  captureImageWrap: {
    width: '100%',
    height: 190,
    borderRadius: RADIUS.MD,
    overflow: 'hidden',
    backgroundColor: '#101a10',
    alignItems: 'center',
    justifyContent: 'center',
  },
  captureImage: {
    width: '100%',
    height: '100%',
  },
  captureImageState: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: 'rgba(16,26,16,0.92)',
  },
  captureImageStateText: {
    fontSize: TYPOGRAPHY.SM,
    color: 'rgba(255,255,255,0.75)',
    fontWeight: '600',
  },
  captureCaption: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: '600',
    marginTop: 8,
    marginHorizontal: 2,
    lineHeight: 16,
  },
  captureEmpty: {
    alignItems: 'center',
    paddingVertical: 22,
    paddingHorizontal: 20,
    borderRadius: RADIUS.LG,
  },
  captureEmptyIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#E8F5E9',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
  },
  captureEmptyTitle: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
    color: COLORS.TEXT_PRIMARY,
    marginBottom: 4,
  },
  captureEmptyText: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    textAlign: 'center',
    lineHeight: 17,
  },
  captureRetryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: COLORS.DANGER,
    borderRadius: RADIUS.SM,
    paddingHorizontal: 12,
    paddingVertical: 7,
    marginTop: 4,
  },
  captureRetryText: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    color: COLORS.WHITE,
  },

  // ── alarm ─────────────────────────────────────────────────────────────────
  // The siren WebView must stay laid out (a display:none view can be dropped by
  // the platform), so it is parked off-screen at 1x1 instead of hidden.
  alarmWebWrap: {
    position: 'absolute',
    width: 1,
    height: 1,
    top: -10,
    left: -10,
    opacity: 0,
  },
  alarmWeb: {
    width: 1,
    height: 1,
    backgroundColor: 'transparent',
  },
  alarmBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.55)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  alarmCard: {
    width: '100%',
    maxWidth: 420,
    backgroundColor: COLORS.WHITE,
    borderRadius: RADIUS.LG,
    overflow: 'hidden',
    paddingBottom: 14,
  },
  alarmHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
    backgroundColor: COLORS.DANGER,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  alarmHeaderText: {
    flex: 1,
    color: COLORS.WHITE,
    fontSize: TYPOGRAPHY.LG,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  alarmLead: {
    fontSize: TYPOGRAPHY.MD,
    color: COLORS.TEXT_PRIMARY,
    fontWeight: '600',
    paddingHorizontal: 14,
    paddingTop: 12,
    paddingBottom: 10,
  },
  alarmFacts: {
    flexDirection: 'row',
    paddingHorizontal: 14,
    paddingTop: 12,
    gap: 10,
  },
  alarmFact: {
    flex: 1,
  },
  alarmFactLabel: {
    fontSize: 10,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: '700',
    letterSpacing: 0.6,
    marginBottom: 2,
  },
  alarmFactValue: {
    fontSize: TYPOGRAPHY.LG,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
  },
  alarmMeta: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    paddingHorizontal: 14,
    marginTop: 8,
    lineHeight: 16,
  },
  alarmButtons: {
    flexDirection: 'row',
    gap: 10,
    paddingHorizontal: 14,
    marginTop: 14,
  },
  alarmBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    borderRadius: RADIUS.MD,
    paddingVertical: 11,
  },
  alarmBtnGhost: {
    backgroundColor: 'rgba(0,0,0,0.06)',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.08)',
  },
  alarmBtnText: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
    color: COLORS.WHITE,
  },
  alarmFootnote: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    textAlign: 'center',
    paddingHorizontal: 14,
    marginTop: 10,
  },
});
