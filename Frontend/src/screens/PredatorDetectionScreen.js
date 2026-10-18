import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  Animated,
  Dimensions,
  Image,
  RefreshControl,
} from 'react-native';
import { WebView } from 'react-native-webview';
import { BarChart } from 'react-native-chart-kit';
import { Volume2, VolumeX } from 'lucide-react-native';
import { COLORS, SHADOW_SM, TYPOGRAPHY, RADIUS } from '../../theme';
import {
  fetchPredatorStatus,
  fetchPredatorAlerts,
  fetchPredatorSummary,
  fetchPredatorDaily,
  pushPredatorAlert,
  predatorStreamUrl,
  predatorImageUrl,
} from '../services/predatorApi';
import { PREDATOR_API_BASE_URL } from '../config/api';
import { usePredatorAlarm } from '../components/PredatorAlarmProvider';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CHART_WIDTH = SCREEN_WIDTH - 48;

const STATUS_POLL_MS = 3000;
const LOG_POLL_MS = 8000;
const STREAM_RETRY_MS = 5000;
const LOG_LIMIT = 12;
const TREND_DAYS = 7;

// Local dark theme — applied to this screen only
const DARK = {
  BG: '#080e08',
  SURFACE: '#0f1a0f',
  CARD: 'rgba(255,255,255,0.06)',
  CARD_BORDER: 'rgba(255,255,255,0.1)',
  TEXT: '#F0F0F0',
  TEXT_SECONDARY: '#888888',
};

// Matches the BGR box colours the backend draws on the stream.
const SPECIES_COLOR = { snake: '#68D89A', scorpion: '#E8A33D', both: COLORS.DANGER };
const SPECIES_ICON = { snake: '🐍', scorpion: '🦂', both: '⚠️' };

const CHART_CONFIG = {
  backgroundColor: 'transparent',
  backgroundGradientFrom: DARK.SURFACE,
  backgroundGradientTo: DARK.BG,
  decimalPlaces: 0,
  color: (opacity = 1) => `rgba(255, 90, 70, ${opacity})`,
  labelColor: () => DARK.TEXT_SECONDARY,
  propsForBackgroundLines: { stroke: 'rgba(255,255,255,0.05)' },
};

function formatClock(iso) {
  if (!iso) return '--:--';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '--:--';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatWhen(iso) {
  if (!iso) return '-';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  return date.toDateString() === new Date().toDateString()
    ? `Today, ${time}`
    : `${date.toLocaleDateString()}, ${time}`;
}

function streamHtml(url) {
  return `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html,body{margin:0;padding:0;background:#000;height:100%;overflow:hidden}
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

export default function PredatorDetectionScreen({ navigation }) {
  const alertAnim = useRef(new Animated.Value(1)).current;

  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const [detections, setDetections] = useState([]);
  const [summary, setSummary] = useState(null);
  const [trend, setTrend] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [cameraHint, setCameraHint] = useState(null);

  const [speciesFilter, setSpeciesFilter] = useState('all');
  const [dismissedAlertId, setDismissedAlertId] = useState(null);
  const [streamNonce, setStreamNonce] = useState(0);
  const [streamError, setStreamError] = useState(false);

  const [busyAction, setBusyAction] = useState(null);

  // ── danger alarm ───────────────────────────────────────────────────────────
  // The siren itself is owned by PredatorAlarmProvider, mounted above the
  // navigator, so that a snake found while the farmer is on another screen
  // still sounds. It polls the backend itself; this screen only shares its
  // mute switch.
  const { muted: sirenMuted, toggleMute } = usePredatorAlarm();

  const retryTimerRef = useRef(null);

  const latestAlert = status && status.latest_alert;
  const alertVisible = Boolean(latestAlert) && latestAlert.id !== dismissedAlertId;
  const cameraLive = Boolean(status && status.stream_connected);
  const latestAlertImage =
    latestAlert && latestAlert.detections && latestAlert.detections.length
      ? predatorImageUrl(latestAlert.detections[0].image_path)
      : null;

  // ── data loading ───────────────────────────────────────────────────────────
  const loadStatus = useCallback(async () => {
    try {
      const data = await fetchPredatorStatus();
      setStatus(data);
      setStatusError(null);
    } catch (error) {
      setStatusError(error.message || 'Backend unreachable');
    }
  }, []);

  // The intrusion log is the alert history: /api/alerts is the only endpoint
  // that hands back individual events with a species, a timestamp and the box
  // that caused them. The CSV behind /api/stats/* has no per-row route.
  const loadLog = useCallback(async () => {
    const [alertData, summaryData, trendData] = await Promise.allSettled([
      fetchPredatorAlerts(LOG_LIMIT),
      fetchPredatorSummary(),
      fetchPredatorDaily(TREND_DAYS),
    ]);
    // Each panel keeps its last good value if only one of the three fails;
    // loadStatus is what surfaces an outright connectivity problem.
    if (alertData.status === 'fulfilled') setDetections(alertData.value.detections);
    if (summaryData.status === 'fulfilled') setSummary(summaryData.value);
    if (trendData.status === 'fulfilled') setTrend(trendData.value);
  }, []);

  useEffect(() => {
    loadStatus();
    loadLog();
    const statusTimer = setInterval(loadStatus, STATUS_POLL_MS);
    const logTimer = setInterval(loadLog, LOG_POLL_MS);
    return () => {
      clearInterval(statusTimer);
      clearInterval(logTimer);
    };
  }, [loadStatus, loadLog]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await Promise.all([loadStatus(), loadLog()]);
    setRefreshing(false);
  }, [loadStatus, loadLog]);

  // ── alert banner pulse ─────────────────────────────────────────────────────
  useEffect(() => {
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(alertAnim, { toValue: 0.25, duration: 500, useNativeDriver: true }),
        Animated.timing(alertAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
      ])
    );
    if (alertVisible) {
      anim.start();
    } else {
      alertAnim.setValue(1);
    }
    return () => anim.stop();
  }, [alertVisible, alertAnim]);

  // ── stream retry ───────────────────────────────────────────────────────────
  const handleStreamMessage = useCallback(event => {
    const message = event.nativeEvent.data;
    if (message === 'error') {
      setStreamError(true);
      if (!retryTimerRef.current) {
        retryTimerRef.current = setTimeout(() => {
          retryTimerRef.current = null;
          setStreamNonce(n => n + 1);
          setStreamError(false);
        }, STREAM_RETRY_MS);
      }
    } else if (message === 'ok') {
      setStreamError(false);
    }
  }, []);

  useEffect(() => () => {
    if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
  }, []);

  // ── actions ────────────────────────────────────────────────────────────────
  const raiseAlarm = useCallback(async () => {
    setBusyAction('alarm');
    try {
      await pushPredatorAlert('both', 'mobile-panic-button');
      await loadStatus();
      setDismissedAlertId(null);
    } catch (error) {
      setStatusError(error.message);
    } finally {
      setBusyAction(null);
    }
  }, [loadStatus]);

  // The RTSP reader is owned by the notebook (start_all()); there is no HTTP
  // route that attaches a camera. So this re-probes the service and remounts
  // the MJPEG view, and says plainly what to do if it is still not streaming.
  const attachCamera = useCallback(async () => {
    setBusyAction('camera');
    setCameraHint(null);
    try {
      const fresh = await fetchPredatorStatus();
      setStatus(fresh);
      setStatusError(null);
      setStreamNonce(n => n + 1);
      setStreamError(false);
      if (!fresh.stream_connected) {
        setCameraHint(
          fresh.stream_error
            ? `Backend is up but the camera is not: ${fresh.stream_error}`
            : 'Backend is up but no camera is attached. Run the start_all() cell in snake_scorpion_rtsp_detection.ipynb.',
        );
      }
    } catch (error) {
      setStatusError(error.message);
    } finally {
      setBusyAction(null);
    }
  }, []);

  // ── derived values ─────────────────────────────────────────────────────────
  // /api/status has no counters; the totals live in /api/stats/summary.
  const todayTotals = summary ? summary.today : { snake: 0, scorpion: 0, total: 0 };
  const allTime = summary ? summary.all_time : { snake: 0, scorpion: 0, total: 0 };

  const cooldown = (status && status.cooldown) || {};
  const suppressed =
    (cooldown.snake ? cooldown.snake.suppressed : 0) +
    (cooldown.scorpion ? cooldown.scorpion.suppressed : 0);

  const filteredDetections =
    speciesFilter === 'all'
      ? detections
      : detections.filter(d => d.species === speciesFilter);

  const trendData = {
    labels: trend ? trend.buckets.map(b => b.label) : [],
    datasets: [{ data: trend ? trend.buckets.map(b => b.snake + b.scorpion) : [] }],
  };
  const hasTrend = trendData.datasets[0].data.some(v => v > 0);

  const speciesShare = species => {
    const total = allTime.snake + allTime.scorpion;
    return total ? Math.round((allTime[species] / total) * 100) : 0;
  };

  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: DARK.BG }]} edges={['bottom']}>
      <View style={styles.darkHeader}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
          <Text style={styles.backText}>← Back</Text>
        </TouchableOpacity>
        <View style={styles.darkHeaderCenter}>
          <View
            style={[
              styles.threatDot,
              { backgroundColor: statusError ? COLORS.DANGER : cameraLive ? SPECIES_COLOR.snake : COLORS.WARNING },
            ]}
          />
          <Text style={styles.darkHeaderTitle}>Predator Detection</Text>
        </View>
        <View style={styles.darkHeaderRight}>
          {/* Pre-mute the siren before a detection ever happens. Same session
              preference the alarm card's Mute button flips. */}
          <TouchableOpacity
            onPress={toggleMute}
            style={styles.muteChip}
            hitSlop={10}
            activeOpacity={0.7}
          >
            {sirenMuted ? (
              <VolumeX size={14} color={COLORS.WARNING} strokeWidth={2.2} />
            ) : (
              <Volume2 size={14} color={SPECIES_COLOR.snake} strokeWidth={2.2} />
            )}
          </TouchableOpacity>
          <Text style={styles.darkHeaderSub}>
            {statusError ? 'OFFLINE' : cameraLive ? `${status.fps} fps` : 'NO CAM'}
          </Text>
        </View>
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={DARK.TEXT_SECONDARY} />
        }
      >
        {/* Backend unreachable */}
        {statusError && (
          <View style={styles.errorBanner}>
            <Text style={styles.errorTitle}>Detection backend unreachable</Text>
            <Text style={styles.errorText}>{statusError}</Text>
            <Text style={styles.errorHint}>
              Run every cell of Backend/Snake&Scropiens Detection/snake_scorpion_rtsp_detection.ipynb
              up to start_all(), then check PREDATOR_API_BASE_URL ({PREDATOR_API_BASE_URL}).
            </Text>
          </View>
        )}

        {/* Live alert from the backend's 15 s-cooldown alert engine */}
        {alertVisible && (
          <Animated.View
            style={[
              styles.predatorBanner,
              { opacity: alertAnim, backgroundColor: latestAlert.severity === 'critical' ? '#8B0000' : COLORS.DANGER },
            ]}
          >
            <Text style={styles.predatorBannerText}>
              {SPECIES_ICON[latestAlert.species] || '⚠️'}  {latestAlert.message}
            </Text>
            <Text style={styles.predatorBannerSub}>
              {formatWhen(latestAlert.timestamp)} • source: {latestAlert.source}
              {latestAlert.detections && latestAlert.detections.length
                ? ` • ${Math.round(latestAlert.detections[0].confidence * 100)}% confidence`
                : ''}
            </Text>
          </Animated.View>
        )}

        {/* Live camera feed — MJPEG with the backend's boxes already drawn */}
        <View style={styles.cameraCard}>
          {cameraLive && !streamError ? (
            <WebView
              key={streamNonce}
              source={{ html: streamHtml(predatorStreamUrl(streamNonce)) }}
              style={styles.webview}
              containerStyle={styles.webviewContainer}
              onMessage={handleStreamMessage}
              scrollEnabled={false}
              javaScriptEnabled
              originWhitelist={['*']}
              mixedContentMode="always"
            />
          ) : (
            <View style={styles.cameraPlaceholder}>
              <Text style={styles.cameraPlaceholderIcon}>📷</Text>
              <Text style={styles.cameraPlaceholderTitle}>
                {streamError ? 'Stream dropped — retrying' : 'No camera attached'}
              </Text>
              <Text style={styles.cameraPlaceholderText}>
                {cameraHint ||
                  'The RTSP reader is started by start_all() in the detector notebook. Tap below to re-check.'}
              </Text>
              <TouchableOpacity
                style={styles.attachBtn}
                onPress={attachCamera}
                disabled={busyAction === 'camera'}
                activeOpacity={0.8}
              >
                <Text style={styles.attachBtnText}>
                  {busyAction === 'camera' ? 'Checking…' : 'Attach camera'}
                </Text>
              </TouchableOpacity>
            </View>
          )}

          {cameraLive && (
            <View style={styles.liveTag}>
              <View style={styles.liveDot} />
              <Text style={styles.liveTagText}>LIVE • {status.processed} frames analysed</Text>
            </View>
          )}
        </View>

        {/* What the filters are doing right now */}
        <View style={styles.statRow}>
          <View style={styles.statCard}>
            <Text style={[styles.statValue, { color: SPECIES_COLOR.snake }]}>{todayTotals.snake}</Text>
            <Text style={styles.statLabel}>Snake today</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={[styles.statValue, { color: SPECIES_COLOR.scorpion }]}>{todayTotals.scorpion}</Text>
            <Text style={styles.statLabel}>Scorpion today</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={[styles.statValue, { color: DARK.TEXT }]}>
              {status ? status.rejected_boxes : 0}
            </Text>
            <Text style={styles.statLabel}>Boxes filtered</Text>
          </View>
        </View>

        {/* Detection trend — real counts from detections_log.csv */}
        <View style={styles.darkCard}>
          <Text style={styles.darkCardTitle}>Detections — last {TREND_DAYS} days</Text>
          {hasTrend ? (
            <BarChart
              data={trendData}
              width={CHART_WIDTH}
              height={170}
              yAxisLabel=""
              yAxisSuffix=""
              chartConfig={CHART_CONFIG}
              fromZero
              style={styles.barChart}
              showValuesOnTopOfBars
            />
          ) : (
            <Text style={styles.cardHint}>No detections logged in this window yet.</Text>
          )}
          <Text style={styles.cardFootnote}>{trend ? trend.note : ''}</Text>
        </View>

        {/* Species split, all time */}
        <View style={styles.darkCard}>
          <Text style={styles.darkCardTitle}>Species split — all time</Text>
          {['snake', 'scorpion'].map(species => (
            <View key={species} style={styles.splitRow}>
              <Text style={[styles.splitLabel, { color: SPECIES_COLOR[species] }]}>
                {SPECIES_ICON[species]} {species}
              </Text>
              <View style={styles.splitBarBg}>
                <View
                  style={[
                    styles.splitBarFill,
                    { width: `${speciesShare(species)}%`, backgroundColor: SPECIES_COLOR[species] },
                  ]}
                />
              </View>
              <Text style={styles.splitValue}>{allTime[species]}</Text>
            </View>
          ))}
          <Text style={styles.cardFootnote}>
            {allTime.total} logged events{summary && summary.last_detection
              ? ` • last at ${formatWhen(summary.last_detection)}`
              : ''}
          </Text>
        </View>

        {/* Alert throttling */}
        <View style={styles.darkCard}>
          <Text style={styles.darkCardTitle}>Alert cooldown</Text>
          {['snake', 'scorpion'].map(species => {
            const c = cooldown[species] || { left: 0, total: 15, suppressed: 0 };
            const pct = c.total ? Math.round((c.left / c.total) * 100) : 0;
            return (
              <View key={species} style={styles.splitRow}>
                <Text style={[styles.splitLabel, { color: SPECIES_COLOR[species] }]}>{species}</Text>
                <View style={styles.splitBarBg}>
                  <View
                    style={[styles.splitBarFill, { width: `${pct}%`, backgroundColor: SPECIES_COLOR[species] }]}
                  />
                </View>
                <Text style={styles.splitValue}>{c.left > 0 ? `${c.left.toFixed(0)}s` : 'ready'}</Text>
              </View>
            );
          })}
          <Text style={styles.cardFootnote}>
            {suppressed} repeat alert{suppressed === 1 ? '' : 's'} throttled by the{' '}
            {cooldown.snake ? cooldown.snake.total : 15}s cooldown.
          </Text>
        </View>

        {/* Intrusion log — straight from the CSV the detector writes */}
        <View style={[styles.darkCard, { paddingBottom: 8 }]}>
          <View style={styles.logHeaderRow}>
            <Text style={[styles.darkCardTitle, { marginBottom: 0 }]}>Intrusion log</Text>
            <View style={styles.filterRow}>
              {['all', 'snake', 'scorpion'].map(f => (
                <TouchableOpacity
                  key={f}
                  onPress={() => setSpeciesFilter(f)}
                  style={[
                    styles.filterChip,
                    speciesFilter === f && { backgroundColor: COLORS.PRIMARY, borderColor: COLORS.PRIMARY_LIGHT },
                  ]}
                  activeOpacity={0.8}
                >
                  <Text
                    style={[
                      styles.filterChipText,
                      speciesFilter === f && { color: COLORS.WHITE },
                    ]}
                  >
                    {f}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {filteredDetections.length === 0 && (
            <Text style={styles.cardHint}>Nothing logged yet.</Text>
          )}

          {filteredDetections.map((item, index) => (
            <View key={item.id || `${item.timestamp}-${index}`} style={styles.logItem}>
              <View style={[styles.logDot, { backgroundColor: SPECIES_COLOR[item.species] }]} />
              <View style={styles.logContent}>
                <View style={styles.logRow}>
                  <Text style={styles.logThreat}>
                    {SPECIES_ICON[item.species]} {item.species}
                  </Text>
                  <Text
                    style={[
                      styles.logConf,
                      { color: item.confidence > 0.8 ? COLORS.DANGER : COLORS.WARNING },
                    ]}
                  >
                    {/* null on a hand-raised alert: it carries no boxes */}
                    {item.confidence == null ? '—' : `${Math.round(item.confidence * 100)}%`}
                  </Text>
                </View>
                <Text style={styles.logDetail}>
                  {formatWhen(item.timestamp)}
                  {item.similarity !== null && item.similarity !== undefined
                    ? ` • sim ${item.similarity.toFixed(2)}`
                    : ''}
                  {item.source && item.source !== 'camera' ? ` • ${item.source}` : ''}
                </Text>
              </View>
            </View>
          ))}
        </View>

        {/* Actions */}
        <View style={styles.actionGrid}>
          <TouchableOpacity
            style={[styles.actionBtn, { backgroundColor: COLORS.DANGER }]}
            onPress={raiseAlarm}
            disabled={busyAction === 'alarm'}
            activeOpacity={0.8}
          >
            <Text style={styles.actionBtnText}>
              {busyAction === 'alarm' ? 'Raising…' : '🚨 Raise alarm'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionBtn, { backgroundColor: COLORS.PRIMARY }]}
            onPress={onRefresh}
            activeOpacity={0.8}
          >
            <Text style={styles.actionBtnText}>↻ Refresh</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionBtn, { backgroundColor: DARK.CARD, borderWidth: 1, borderColor: DARK.CARD_BORDER }]}
            onPress={() => latestAlert && setDismissedAlertId(latestAlert.id)}
            disabled={!alertVisible}
            activeOpacity={0.8}
          >
            <Text style={[styles.actionBtnText, { color: DARK.TEXT_SECONDARY }]}>✓ Dismiss alert</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionBtn, { backgroundColor: DARK.CARD, borderWidth: 1, borderColor: DARK.CARD_BORDER }]}
            onPress={() => {
              setStreamNonce(n => n + 1);
              setStreamError(false);
            }}
            activeOpacity={0.8}
          >
            <Text style={[styles.actionBtnText, { color: DARK.TEXT_SECONDARY }]}>🔄 Reload feed</Text>
          </TouchableOpacity>
        </View>

        {/* The alert's own `snapshot` lives in detection/alerts, which no route
            serves. The annotated frame that caused it sits in detection/<species>,
            which /captures/{species}/{name} does serve — so show that one. */}
        {latestAlertImage && (
          <View style={styles.darkCard}>
            <Text style={styles.darkCardTitle}>Last alert snapshot</Text>
            <Image
              source={{ uri: latestAlertImage }}
              style={styles.snapshot}
              resizeMode="contain"
            />
            <Text style={styles.cardFootnote}>{formatClock(latestAlert.timestamp)}</Text>
          </View>
        )}
      </ScrollView>
      {/* The siren and its emergency card render from PredatorAlarmProvider,
          above the navigator - see the danger alarm note above. */}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
  },
  darkHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    paddingTop: 16,
    backgroundColor: DARK.SURFACE,
    borderBottomWidth: 1,
    borderBottomColor: DARK.CARD_BORDER,
  },
  backBtn: {
    minWidth: 60,
  },
  backText: {
    color: COLORS.PRIMARY_LIGHT,
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '600',
  },
  darkHeaderCenter: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  threatDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  darkHeaderTitle: {
    color: DARK.TEXT,
    fontSize: TYPOGRAPHY.LG,
    fontWeight: '700',
  },
  darkHeaderRight: {
    minWidth: 60,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 8,
  },
  muteChip: {
    padding: 4,
    borderRadius: RADIUS.SM,
    backgroundColor: DARK.CARD,
    borderWidth: 1,
    borderColor: DARK.CARD_BORDER,
  },
  darkHeaderSub: {
    color: DARK.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.SM,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 32,
    paddingTop: 8,
  },
  errorBanner: {
    marginHorizontal: 16,
    marginBottom: 12,
    padding: 12,
    borderRadius: RADIUS.MD,
    backgroundColor: 'rgba(211,47,47,0.15)',
    borderWidth: 1,
    borderColor: 'rgba(211,47,47,0.4)',
  },
  errorTitle: {
    color: COLORS.DANGER,
    fontWeight: '800',
    fontSize: TYPOGRAPHY.MD,
  },
  errorText: {
    color: DARK.TEXT,
    fontSize: TYPOGRAPHY.SM,
    marginTop: 3,
  },
  errorHint: {
    color: DARK.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.SM,
    marginTop: 6,
    lineHeight: 16,
  },
  predatorBanner: {
    marginHorizontal: 16,
    marginBottom: 12,
    borderRadius: RADIUS.MD,
    paddingVertical: 14,
    paddingHorizontal: 12,
    alignItems: 'center',
    ...SHADOW_SM,
  },
  predatorBannerText: {
    color: DARK.TEXT,
    fontWeight: '900',
    fontSize: TYPOGRAPHY.LG,
    letterSpacing: 0.6,
    textAlign: 'center',
  },
  predatorBannerSub: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: TYPOGRAPHY.SM,
    marginTop: 4,
    textAlign: 'center',
  },
  cameraCard: {
    marginHorizontal: 16,
    height: 240,
    borderRadius: RADIUS.LG,
    overflow: 'hidden',
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: DARK.CARD_BORDER,
  },
  webview: {
    flex: 1,
    backgroundColor: '#000',
  },
  webviewContainer: {
    backgroundColor: '#000',
  },
  cameraPlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    backgroundColor: '#111a11',
  },
  cameraPlaceholderIcon: {
    fontSize: 28,
    marginBottom: 6,
  },
  cameraPlaceholderTitle: {
    color: DARK.TEXT,
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
  },
  cameraPlaceholderText: {
    color: DARK.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.SM,
    textAlign: 'center',
    marginTop: 6,
    lineHeight: 16,
  },
  attachBtn: {
    marginTop: 12,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: RADIUS.MD,
    backgroundColor: COLORS.PRIMARY,
  },
  attachBtnText: {
    color: COLORS.WHITE,
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
  },
  liveTag: {
    position: 'absolute',
    top: 10,
    left: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: RADIUS.SM,
  },
  liveDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: SPECIES_COLOR.snake,
  },
  liveTagText: {
    color: DARK.TEXT,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.8,
  },
  statRow: {
    flexDirection: 'row',
    marginHorizontal: 16,
    marginTop: 12,
    gap: 10,
  },
  statCard: {
    flex: 1,
    backgroundColor: DARK.CARD,
    borderRadius: RADIUS.MD,
    borderWidth: 1,
    borderColor: DARK.CARD_BORDER,
    paddingVertical: 12,
    alignItems: 'center',
  },
  statValue: {
    fontSize: TYPOGRAPHY.XXL,
    fontWeight: '800',
  },
  statLabel: {
    color: DARK.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.SM,
    marginTop: 2,
  },
  darkCard: {
    marginHorizontal: 16,
    marginTop: 12,
    backgroundColor: DARK.CARD,
    borderRadius: RADIUS.LG,
    padding: 14,
    borderWidth: 1,
    borderColor: DARK.CARD_BORDER,
  },
  darkCardTitle: {
    color: DARK.TEXT,
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
    marginBottom: 10,
    letterSpacing: 0.2,
  },
  cardHint: {
    color: DARK.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.SM,
    lineHeight: 16,
  },
  cardFootnote: {
    color: DARK.TEXT_SECONDARY,
    fontSize: 10,
    marginTop: 8,
    lineHeight: 14,
  },
  barChart: {
    borderRadius: RADIUS.MD,
    marginLeft: -8,
  },
  splitRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  splitLabel: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    width: 80,
  },
  splitBarBg: {
    flex: 1,
    height: 12,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 6,
    overflow: 'hidden',
  },
  splitBarFill: {
    height: '100%',
    borderRadius: 6,
  },
  splitValue: {
    color: DARK.TEXT,
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    minWidth: 44,
    textAlign: 'right',
  },
  logHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  filterRow: {
    flexDirection: 'row',
    gap: 6,
  },
  filterChip: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: RADIUS.SM,
    borderWidth: 1,
    borderColor: DARK.CARD_BORDER,
  },
  filterChipText: {
    color: DARK.TEXT_SECONDARY,
    fontSize: 10,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  logItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
    gap: 10,
  },
  logDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginTop: 4,
  },
  logContent: {
    flex: 1,
  },
  logRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  logThreat: {
    color: DARK.TEXT,
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  logConf: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
  },
  logDetail: {
    color: DARK.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.SM,
    marginTop: 2,
  },
  actionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: 16,
    marginTop: 14,
    gap: 10,
  },
  actionBtn: {
    flex: 1,
    minWidth: '45%',
    paddingVertical: 13,
    paddingHorizontal: 10,
    borderRadius: RADIUS.MD,
    alignItems: 'center',
  },
  actionBtnText: {
    color: DARK.TEXT,
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    textAlign: 'center',
  },
  snapshot: {
    width: '100%',
    height: 180,
    borderRadius: RADIUS.MD,
    backgroundColor: '#000',
  },
});
