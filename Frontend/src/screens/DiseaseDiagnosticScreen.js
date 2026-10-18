import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  Image,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
  SafeAreaView,
  Dimensions,
  Animated,
} from 'react-native';
import { Svg, Rect } from 'react-native-svg';
import {
  Stethoscope,
  Scan,
  Camera,
  Images,
  ClipboardList,
  Activity,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Eye,
  RefreshCw,
  Zap,
  ShieldCheck,
  History,
  Wifi,
  WifiOff,
  ImageOff,
} from 'lucide-react-native';
import Header from '../components/Header';
import { COLORS, SHADOW, SHADOW_SM, TYPOGRAPHY, RADIUS } from '../../theme';
import { DISEASE_API_BASE_URL } from '../config/api';
import {
  analyzeImage,
  checkDiseaseApiHealth,
  gradcamImageUrl,
  DiseaseApiError,
  isDiseaseApiError,
  ERROR_CODE,
  SCAN_TYPE,
} from '../services/diseaseApi';
import { captureFromCamera, pickFromGallery } from '../services/imagePicker';

// ─── Dimensions ───────────────────────────────────────────────────────────────
const { width: SCREEN_W } = Dimensions.get('window');
const VF_W = SCREEN_W - 32;         // viewfinder width (16 px page padding each side)
const VF_H = 260;                   // viewfinder height
const CARD_INNER_W = SCREEN_W - 60; // card inner content width
const COMP_W = Math.floor((CARD_INNER_W - 34) / 2);
const COMP_H = 132;

// Guide-frame SVG constants (relative to viewfinder)
const G_PAD = 30;                   // padding from viewfinder edge to guide rect
const C_LEN = 22;                   // corner bracket arm length

// ─── Phases ───────────────────────────────────────────────────────────────────
const PHASE = { IDLE: 'idle', ANALYZING: 'analyzing', RESULTS: 'results' };

// ─── Backend vocabulary ───────────────────────────────────────────────────────
// Class names come straight from Backend/Disease_detection/app/main.py
// (FECAL_CLASSES / BODY_CLASSES). Everything below is presentation only.
const SCAN_MODES = [
  {
    key: SCAN_TYPE.FECAL,
    emoji: '🔬',
    label: 'Fecal Sample',
    detects: 'Coccidiosis · Newcastle · Healthy',
    tip: 'Align the dropping inside the frame, filling most of it.',
  },
  {
    key: SCAN_TYPE.BODY,
    emoji: '🐔',
    label: 'Bird Body',
    detects: 'Fowlpox · Healthy',
    tip: 'Frame the bird’s head and comb clearly, in good light.',
  },
];

const DISEASE_INFO = {
  Newcastle: {
    subtitle: 'Avian paramyxovirus type 1 (NDV)',
    healthy: false,
  },
  Coccidiosis: {
    subtitle: 'Intestinal Eimeria spp. infection',
    healthy: false,
  },
  Fowlpox: {
    subtitle: 'Avipoxvirus — cutaneous form',
    healthy: false,
  },
  Healthy: {
    subtitle: 'No disease indicators found in this sample',
    healthy: true,
  },
};

const RISK_STYLE = {
  High: { color: '#E53935', label: 'HIGH RISK' },
  Medium: { color: '#F57C00', label: 'MODERATE' },
  Low: { color: '#2E7D32', label: 'LOW RISK' },
};

const MODEL_TAG = 'EfficientNetB0 disease models · MobileNetV2 input validator';

// ─── Helpers ──────────────────────────────────────────────────────────────────
function modeFor(key) {
  return SCAN_MODES.find(m => m.key === key) || SCAN_MODES[0];
}

function diseaseInfoFor(prediction) {
  return (
    DISEASE_INFO[prediction] || {
      subtitle: 'Unrecognised class returned by the model',
      healthy: false,
    }
  );
}

function riskStyleFor(level) {
  return RISK_STYLE[level] || RISK_STYLE.Medium;
}

function accentFor(result) {
  if (!result) return COLORS.PRIMARY;
  if (diseaseInfoFor(result.prediction).healthy) return '#2E7D32';
  return riskStyleFor(result.risk_level).color;
}

function formatClock(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function RiskBadge({ level }) {
  const style = riskStyleFor(level);
  const Icon = level === 'Low' ? ShieldCheck : AlertTriangle;
  return (
    <View style={[ss.riskBadge, { backgroundColor: style.color }]}>
      <Icon size={11} color="#fff" strokeWidth={2.5} />
      <Text style={ss.riskBadgeText}>{style.label}</Text>
    </View>
  );
}

/** Animated probability bar. `percent` is 0-100, straight from the backend. */
function ConfidenceBar({ label, percent, highlight }) {
  const barAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(barAnim, {
      toValue: Math.max(0, Math.min(100, percent)) / 100,
      duration: 900,
      useNativeDriver: false,
    }).start();
  }, [percent, barAnim]);

  const barW = barAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  const fillColor = highlight
    ? percent >= 85
      ? '#2E7D32'
      : percent >= 70
      ? '#F57C00'
      : '#E53935'
    : 'rgba(27,94,32,0.35)';

  return (
    <View style={ss.confRow}>
      <View style={ss.confMeta}>
        <Text style={[ss.confLabel, highlight && ss.confLabelStrong]}>{label}</Text>
        <Text style={ss.confPct}>{percent.toFixed(1)}%</Text>
      </View>
      <View style={ss.confTrack}>
        <Animated.View
          style={[ss.confFill, { width: barW, backgroundColor: fillColor }]}
        />
      </View>
    </View>
  );
}

/** Corner brackets + guide rectangle drawn over the viewfinder. */
function GuideOverlay({ active, showCrosshair }) {
  const stroke = active ? '#00FF41' : 'rgba(0,255,65,0.55)';
  return (
    <Svg style={StyleSheet.absoluteFill} width={VF_W} height={VF_H}>
      <Rect
        x={G_PAD}
        y={G_PAD}
        width={VF_W - G_PAD * 2}
        height={VF_H - G_PAD * 2}
        stroke={stroke}
        strokeWidth="1.5"
        fill="transparent"
        strokeDasharray={active ? '0' : '9,6'}
        rx="5"
      />
      {/* top-left */}
      <Rect x={G_PAD - 3} y={G_PAD - 3} width={C_LEN} height={3} fill="#00FF41" rx="1" />
      <Rect x={G_PAD - 3} y={G_PAD - 3} width={3} height={C_LEN} fill="#00FF41" rx="1" />
      {/* top-right */}
      <Rect x={VF_W - G_PAD - C_LEN + 3} y={G_PAD - 3} width={C_LEN} height={3} fill="#00FF41" rx="1" />
      <Rect x={VF_W - G_PAD} y={G_PAD - 3} width={3} height={C_LEN} fill="#00FF41" rx="1" />
      {/* bottom-left */}
      <Rect x={G_PAD - 3} y={VF_H - G_PAD} width={C_LEN} height={3} fill="#00FF41" rx="1" />
      <Rect x={G_PAD - 3} y={VF_H - G_PAD - C_LEN + 3} width={3} height={C_LEN} fill="#00FF41" rx="1" />
      {/* bottom-right */}
      <Rect x={VF_W - G_PAD - C_LEN + 3} y={VF_H - G_PAD} width={C_LEN} height={3} fill="#00FF41" rx="1" />
      <Rect x={VF_W - G_PAD} y={VF_H - G_PAD - C_LEN + 3} width={3} height={C_LEN} fill="#00FF41" rx="1" />
      {showCrosshair && (
        <>
          <Rect x={VF_W / 2 - 14} y={VF_H / 2 - 1} width={28} height={2} fill="rgba(0,255,65,0.35)" rx="1" />
          <Rect x={VF_W / 2 - 1} y={VF_H / 2 - 14} width={2} height={28} fill="rgba(0,255,65,0.35)" rx="1" />
        </>
      )}
    </Svg>
  );
}

/** Remote Grad-CAM image with its own loading / failure states. */
function RemoteImage({ uri, width, height }) {
  const [state, setState] = useState('loading');

  return (
    <View style={[ss.compPhoto, { width, height }]}>
      <Image
        source={{ uri }}
        style={{ width, height }}
        resizeMode="cover"
        onLoad={() => setState('done')}
        onError={() => setState('failed')}
      />
      {state === 'loading' && (
        <View style={ss.remoteOverlay}>
          <ActivityIndicator size="small" color="#00FF41" />
        </View>
      )}
      {state === 'failed' && (
        <View style={ss.remoteOverlay}>
          <ImageOff size={20} color="rgba(255,255,255,0.6)" strokeWidth={1.8} />
          <Text style={ss.remoteFailText}>Could not load</Text>
        </View>
      )}
    </View>
  );
}

// ─── Main Screen ──────────────────────────────────────────────────────────────
export default function DiseaseDiagnosticScreen({ navigation }) {
  const [phase, setPhase] = useState(PHASE.IDLE);
  const [scanType, setScanType] = useState(SCAN_TYPE.FECAL);
  const [asset, setAsset] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState({ state: 'checking', version: null });
  const [history, setHistory] = useState([]);
  const [treatmentOpen, setTreatmentOpen] = useState(true);
  const [completedAt, setCompletedAt] = useState(null);

  const scanLineAnim = useRef(new Animated.Value(0)).current;
  const scanLoopRef = useRef(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      scanLoopRef.current?.stop();
    };
  }, []);

  // ─── Backend reachability ─────────────────────────────────────────────────
  const probeHealth = useCallback(async () => {
    setHealth({ state: 'checking', version: null });
    try {
      const data = await checkDiseaseApiHealth();
      if (!mountedRef.current) return;
      setHealth({ state: 'online', version: data?.version ?? null });
    } catch (e) {
      if (!mountedRef.current) return;
      setHealth({ state: 'offline', version: null });
    }
  }, []);

  useEffect(() => {
    probeHealth();
  }, [probeHealth]);

  // ─── Laser sweep while the request is in flight ───────────────────────────
  useEffect(() => {
    if (phase !== PHASE.ANALYZING) return undefined;

    scanLineAnim.setValue(0);
    scanLoopRef.current = Animated.loop(
      Animated.timing(scanLineAnim, {
        toValue: VF_H - G_PAD * 2,
        duration: 1500,
        useNativeDriver: true,
      }),
    );
    scanLoopRef.current.start();

    return () => scanLoopRef.current?.stop();
  }, [phase, scanLineAnim]);

  // ─── Image selection ──────────────────────────────────────────────────────
  const runPicker = useCallback(async picker => {
    try {
      const { cancelled, asset: picked } = await picker();
      if (cancelled || !mountedRef.current) return;
      setAsset(picked);
      setError(null);
      setResult(null);
    } catch (e) {
      if (!mountedRef.current) return;
      setError(
        new DiseaseApiError(
          'PICKER_ERROR',
          e.message || 'The image could not be opened.',
        ),
      );
    }
  }, []);

  const handleCamera = () => runPicker(captureFromCamera);
  const handleGallery = () => runPicker(pickFromGallery);

  // ─── Analysis ─────────────────────────────────────────────────────────────
  const handleAnalyze = useCallback(async () => {
    if (!asset) return;

    setError(null);
    setPhase(PHASE.ANALYZING);

    try {
      const data = await analyzeImage(scanType, asset);
      if (!mountedRef.current) return;

      const finishedAt = new Date();
      setResult(data);
      setCompletedAt(finishedAt);
      setTreatmentOpen(true);
      setPhase(PHASE.RESULTS);
      setHealth(current =>
        current.state === 'online' ? current : { ...current, state: 'online' },
      );
      setHistory(previous =>
        [
          {
            id: `${finishedAt.getTime()}`,
            at: finishedAt,
            scanType: data.scan_type,
            prediction: data.prediction,
            confidence: data.confidence,
            riskLevel: data.risk_level,
            uri: asset.uri,
          },
          ...previous,
        ].slice(0, 10),
      );
    } catch (e) {
      if (!mountedRef.current) return;
      setError(
        isDiseaseApiError(e)
          ? e
          : new DiseaseApiError(ERROR_CODE.SERVER, e.message || 'Analysis failed.'),
      );
      setPhase(PHASE.IDLE);
      if (isDiseaseApiError(e) && e.code === ERROR_CODE.NETWORK) {
        setHealth({ state: 'offline', version: null });
      }
    }
  }, [asset, scanType]);

  const handleReset = () => {
    scanLineAnim.setValue(0);
    setAsset(null);
    setResult(null);
    setError(null);
    setCompletedAt(null);
    setPhase(PHASE.IDLE);
  };

  // Offered by the WRONG_IMAGE_TYPE error card: the validator already told us
  // which model this image belongs to, so let the user switch in one tap.
  const handleSwitchScanType = detectedType => {
    setScanType(detectedType === 'FECAL' ? SCAN_TYPE.FECAL : SCAN_TYPE.BODY);
    setError(null);
  };

  // ─── Session history ──────────────────────────────────────────────────────
  const renderHistory = () => (
    <View style={[ss.glassCard, SHADOW_SM]}>
      <View style={ss.cardTitleRow}>
        <History size={15} color={COLORS.PRIMARY} strokeWidth={2} />
        <Text style={ss.cardTitle}>Scans This Session</Text>
      </View>
      <Text style={ss.cardSubtitle}>
        Kept in memory only — cleared when the app restarts.
      </Text>
      {history.map(entry => {
        const info = diseaseInfoFor(entry.prediction);
        const color = info.healthy ? '#2E7D32' : riskStyleFor(entry.riskLevel).color;
        return (
          <View key={entry.id} style={ss.historyRow}>
            <Image source={{ uri: entry.uri }} style={ss.historyThumb} />
            <View style={ss.historyBody}>
              <Text style={[ss.historyName, { color }]}>{entry.prediction}</Text>
              <Text style={ss.historyMeta}>
                {entry.scanType} · {entry.confidence}% · {formatClock(entry.at)}
              </Text>
            </View>
            <View style={[ss.historyDot, { backgroundColor: color }]} />
          </View>
        );
      })}
    </View>
  );

  // ─── Error card ───────────────────────────────────────────────────────────
  const renderError = () => {
    if (!error) return null;

    const detail = error.detail || {};
    const isConnectivity =
      error.code === ERROR_CODE.NETWORK || error.code === ERROR_CODE.TIMEOUT;

    return (
      <View style={[ss.errorCard, SHADOW_SM]}>
        <View style={ss.errorHeadRow}>
          <View style={ss.errorIconBg}>
            <AlertTriangle size={16} color="#C62828" strokeWidth={2.2} />
          </View>
          <Text style={ss.errorTitle}>
            {isConnectivity ? 'Backend Unreachable' : 'Image Rejected'}
          </Text>
        </View>

        <Text style={ss.errorMessage}>{error.message}</Text>

        {/* Quality failure — show the exact scores the backend measured. */}
        {error.code === ERROR_CODE.IMAGE_QUALITY_FAILED && detail.image_quality && (
          <View style={ss.errorFacts}>
            {detail.image_quality.issues?.map(issue => (
              <Text key={issue} style={ss.errorIssue}>• {issue}</Text>
            ))}
            <Text style={ss.errorMeta}>
              Sharpness {detail.image_quality.blur_score} (min 80) · Brightness{' '}
              {detail.image_quality.brightness_score} (40–215)
            </Text>
          </View>
        )}

        {/* Validator says this image belongs to the other model. */}
        {error.code === ERROR_CODE.WRONG_IMAGE_TYPE && (
          <View style={ss.errorFacts}>
            <Text style={ss.errorMeta}>
              Detected as {detail.detected_type} ({detail.validator_confidence}% confident)
            </Text>
            <TouchableOpacity
              style={ss.errorActionBtn}
              onPress={() => handleSwitchScanType(detail.detected_type)}
              activeOpacity={0.85}
            >
              <RefreshCw size={13} color="#fff" strokeWidth={2.5} />
              <Text style={ss.errorActionText}>
                Switch to {detail.detected_type === 'FECAL' ? 'Fecal Sample' : 'Bird Body'} scan
              </Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Validator says this is neither a body nor a fecal image. It still
            sends detected_type/validator_confidence, so show them. */}
        {error.code === ERROR_CODE.UNSUPPORTED_IMAGE_TYPE &&
          detail.validator_confidence != null && (
            <Text style={ss.errorMeta}>
              Detected as {detail.detected_type} ({detail.validator_confidence}% confident) —
              neither a bird body nor a fecal sample.
            </Text>
          )}

        {error.code === ERROR_CODE.LOW_CONFIDENCE && detail.confidence != null && (
          <Text style={ss.errorMeta}>
            Top match reached only {detail.confidence}% — below the 70% safety threshold.
          </Text>
        )}

        {isConnectivity && (
          <View style={ss.errorFacts}>
            <Text style={ss.errorMeta}>Target: {DISEASE_API_BASE_URL}</Text>
            <TouchableOpacity
              style={ss.errorActionBtn}
              onPress={probeHealth}
              activeOpacity={0.85}
            >
              <RefreshCw size={13} color="#fff" strokeWidth={2.5} />
              <Text style={ss.errorActionText}>Retry connection</Text>
            </TouchableOpacity>
          </View>
        )}

        {error.action ? <Text style={ss.errorAction}>{error.action}</Text> : null}
      </View>
    );
  };

  // ─── Scanner view (IDLE + ANALYZING) ──────────────────────────────────────
  const renderScanner = () => {
    const isAnalyzing = phase === PHASE.ANALYZING;
    const mode = modeFor(scanType);

    return (
      <ScrollView
        style={ss.scroll}
        contentContainerStyle={ss.scannerContent}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Screen header ── */}
        <View style={ss.screenHeader}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={ss.backBtn}>
            <Text style={ss.backText}>← Back</Text>
          </TouchableOpacity>
          <View style={ss.titleRow}>
            <Stethoscope size={21} color={COLORS.PRIMARY} strokeWidth={1.8} />
            <Text style={ss.screenTitle}>Disease Diagnostic</Text>
          </View>
          <Text style={ss.screenSubtitle}>{MODEL_TAG}</Text>
        </View>

        {/* ── Backend status pill ── */}
        <TouchableOpacity
          style={ss.healthPill}
          onPress={probeHealth}
          activeOpacity={0.8}
          disabled={health.state === 'checking'}
        >
          {health.state === 'checking' ? (
            <ActivityIndicator size="small" color={COLORS.TEXT_SECONDARY} />
          ) : health.state === 'online' ? (
            <Wifi size={13} color={COLORS.PRIMARY} strokeWidth={2.2} />
          ) : (
            <WifiOff size={13} color={COLORS.DANGER} strokeWidth={2.2} />
          )}
          <Text
            style={[
              ss.healthText,
              health.state === 'offline' && { color: COLORS.DANGER },
              health.state === 'online' && { color: COLORS.PRIMARY },
            ]}
          >
            {health.state === 'checking'
              ? 'Contacting AI backend…'
              : health.state === 'online'
              ? `AI backend online${health.version ? ` · v${health.version}` : ''}`
              : 'AI backend offline — tap to retry'}
          </Text>
        </TouchableOpacity>

        {/* ── Scan-mode toggle ── */}
        <View style={ss.toggleCard}>
          {SCAN_MODES.map(opt => (
            <TouchableOpacity
              key={opt.key}
              style={[ss.toggleBtn, scanType === opt.key && ss.toggleBtnActive]}
              onPress={() => {
                if (isAnalyzing) return;
                setScanType(opt.key);
                setError(null);
              }}
              disabled={isAnalyzing}
              activeOpacity={0.8}
            >
              <Text
                style={[
                  ss.toggleBtnText,
                  scanType === opt.key && ss.toggleBtnTextActive,
                ]}
              >
                {opt.emoji}  {opt.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
        <Text style={ss.modeDetects}>Detects: {mode.detects}</Text>

        {/* ── Viewfinder / preview ── */}
        <View style={ss.vfWrapper}>
          <View style={ss.vfBar}>
            <View style={ss.liveBadge}>
              <View
                style={[
                  ss.liveDot,
                  { backgroundColor: isAnalyzing ? '#00FF41' : asset ? '#FFC107' : '#888' },
                ]}
              />
              <Text style={ss.liveText}>
                {isAnalyzing ? 'ANALYZING' : asset ? 'READY TO ANALYZE' : 'NO IMAGE'}
              </Text>
            </View>
            <Text style={ss.vfModeLabel}>{mode.label}</Text>
          </View>

          <View style={ss.vfFeed}>
            {asset ? (
              <Image
                source={{ uri: asset.uri }}
                style={StyleSheet.absoluteFill}
                resizeMode="cover"
              />
            ) : null}

            <GuideOverlay active={isAnalyzing} showCrosshair={!asset} />

            {!asset && (
              <View style={ss.emptyHint}>
                <Camera size={30} color="rgba(0,255,65,0.5)" strokeWidth={1.6} />
                <Text style={ss.emptyHintTitle}>No image selected</Text>
                <Text style={ss.emptyHintText}>{mode.tip}</Text>
              </View>
            )}

            {asset && !isAnalyzing && (
              <View style={ss.alignTip}>
                <Text style={ss.alignTipText}>{mode.tip}</Text>
              </View>
            )}

            {isAnalyzing && (
              <>
                <Animated.View
                  style={[ss.laserLine, { transform: [{ translateY: scanLineAnim }] }]}
                  pointerEvents="none"
                />
                <View style={ss.scanStatusOverlay}>
                  <Scan size={18} color="#00FF41" strokeWidth={2} />
                  <Text style={ss.scanStatusText}>Uploading & classifying…</Text>
                </View>
              </>
            )}
          </View>
        </View>

        {/* ── Source buttons ── */}
        <View style={ss.sourceRow}>
          <TouchableOpacity
            style={[ss.sourceBtn, isAnalyzing && ss.btnDisabled]}
            onPress={handleCamera}
            disabled={isAnalyzing}
            activeOpacity={0.85}
          >
            <Camera size={17} color={COLORS.PRIMARY} strokeWidth={2} />
            <Text style={ss.sourceBtnText}>{asset ? 'Retake' : 'Camera'}</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[ss.sourceBtn, isAnalyzing && ss.btnDisabled]}
            onPress={handleGallery}
            disabled={isAnalyzing}
            activeOpacity={0.85}
          >
            <Images size={17} color={COLORS.PRIMARY} strokeWidth={2} />
            <Text style={ss.sourceBtnText}>Gallery</Text>
          </TouchableOpacity>
        </View>

        {/* ── Analyze ── */}
        <View style={ss.captureSection}>
          {isAnalyzing ? (
            <View style={ss.processingBanner}>
              <ActivityIndicator size="large" color={COLORS.PRIMARY} />
              <Text style={ss.processingTitle}>Processing image…</Text>
              <Text style={ss.processingSubtitle}>
                Quality check → input validation → disease model → Grad-CAM
              </Text>
            </View>
          ) : (
            <TouchableOpacity
              style={ss.captureBtn}
              onPress={handleAnalyze}
              disabled={!asset}
              activeOpacity={0.85}
            >
              <View style={[ss.captureBtnRing, !asset && ss.captureRingDisabled]}>
                <View style={[ss.captureBtnCore, !asset && ss.captureCoreDisabled]}>
                  <Scan size={26} color={COLORS.WHITE} strokeWidth={2} />
                </View>
              </View>
              <Text style={[ss.captureBtnLabel, !asset && ss.captureLabelDisabled]}>
                {asset ? 'Analyze Image' : 'Add an image first'}
              </Text>
            </TouchableOpacity>
          )}
        </View>

        {!isAnalyzing && renderError()}

        {history.length > 0 && !isAnalyzing && renderHistory()}
      </ScrollView>
    );
  };

  // ─── Results view ─────────────────────────────────────────────────────────
  const renderResults = () => {
    const info = diseaseInfoFor(result.prediction);
    const accent = accentFor(result);
    const quality = result.image_quality || {};
    const recommendation = result.recommendation || null;
    const gradcamUri = gradcamImageUrl(result.gradcam_image);

    // Highest probability first, so the diagnosis sits at the top of the list.
    const probabilities = Object.entries(result.probabilities || {}).sort(
      (a, b) => b[1] - a[1],
    );

    return (
      <ScrollView
        style={ss.scroll}
        contentContainerStyle={ss.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Top bar ── */}
        <View style={ss.resultsTopBar}>
          <TouchableOpacity style={ss.newScanBtn} onPress={handleReset} activeOpacity={0.8}>
            <RefreshCw size={13} color={COLORS.PRIMARY} strokeWidth={2.5} />
            <Text style={ss.newScanText}>New Scan</Text>
          </TouchableOpacity>
          <Text style={ss.resultsTitle}>Analysis Results</Text>
          <Text style={ss.resultsTimestamp}>
            {completedAt ? formatClock(completedAt) : ''}
          </Text>
        </View>

        {/* ── Primary diagnosis ── */}
        <View style={[ss.diseaseCard, { borderColor: `${accent}33` }, SHADOW]}>
          <View style={ss.diseaseCardTop}>
            <View style={ss.diseaseCardLeft}>
              <Text style={ss.diagnosisLabel}>
                {info.healthy ? 'Screening Result' : 'Primary Diagnosis'}
              </Text>
              <Text style={[ss.diseaseName, { color: accent }]}>{result.prediction}</Text>
              <Text style={ss.diseaseVirusLabel}>{info.subtitle}</Text>
            </View>
            <View style={[ss.diseaseIconBg, { backgroundColor: `${accent}1A` }]}>
              {info.healthy ? (
                <ShieldCheck size={28} color={accent} strokeWidth={1.7} />
              ) : (
                <Stethoscope size={28} color={accent} strokeWidth={1.7} />
              )}
            </View>
          </View>

          <View style={ss.confidenceHeadline}>
            <Text style={[ss.confidenceValue, { color: accent }]}>
              {result.confidence}%
            </Text>
            <Text style={ss.confidenceCaption}>model confidence</Text>
          </View>

          <View style={ss.diseaseTagRow}>
            <RiskBadge level={result.risk_level} />
            <View style={ss.diseaseTag}>
              <Text style={ss.diseaseTagText}>
                {result.scan_type === 'FECAL' ? '🔬 Fecal Analysis' : '🐔 Body Scan'}
              </Text>
            </View>
            <View style={ss.diseaseTag}>
              <Text style={ss.diseaseTagText}>{result.status}</Text>
            </View>
          </View>
        </View>

        {/* ── Class probabilities ── */}
        <View style={[ss.glassCard, SHADOW_SM]}>
          <View style={ss.cardTitleRow}>
            <Activity size={15} color={COLORS.PRIMARY} strokeWidth={2} />
            <Text style={ss.cardTitle}>Class Probabilities</Text>
          </View>
          <Text style={ss.cardSubtitle}>
            Softmax output across every class this model can predict.
          </Text>
          {probabilities.map(([className, percent], index) => (
            <View key={className}>
              {index > 0 && <View style={ss.confDivider} />}
              <ConfidenceBar
                label={className}
                percent={percent}
                highlight={className === result.prediction}
              />
            </View>
          ))}
          <Text style={ss.modelTag}>{MODEL_TAG}</Text>
        </View>

        {/* ── Grad-CAM ── */}
        <View style={[ss.glassCard, SHADOW_SM]}>
          <View style={ss.cardTitleRow}>
            <Eye size={15} color={COLORS.PRIMARY} strokeWidth={2} />
            <Text style={ss.cardTitle}>AI Attention Map</Text>
          </View>
          <Text style={ss.cardSubtitle}>
            {gradcamUri
              ? 'Grad-CAM overlay — warmer regions drove the diagnosis.'
              : info.healthy
              ? 'The backend only renders a heatmap when a disease is detected.'
              : 'The heatmap could not be generated for this scan.'}
          </Text>

          <View style={ss.compRow}>
            <View style={ss.compCol}>
              {asset ? (
                <Image
                  source={{ uri: asset.uri }}
                  style={[ss.compPhoto, { width: COMP_W, height: COMP_H }]}
                  resizeMode="cover"
                />
              ) : (
                <View style={[ss.compPhoto, ss.compEmpty, { width: COMP_W, height: COMP_H }]} />
              )}
              <Text style={ss.compCaption}>Your sample</Text>
            </View>

            <View style={ss.compCol}>
              {gradcamUri ? (
                <RemoteImage key={gradcamUri} uri={gradcamUri} width={COMP_W} height={COMP_H} />
              ) : (
                <View style={[ss.compPhoto, ss.compEmpty, { width: COMP_W, height: COMP_H }]}>
                  <ImageOff size={22} color="rgba(255,255,255,0.5)" strokeWidth={1.6} />
                </View>
              )}
              <Text style={ss.compCaption}>AI attention</Text>
            </View>
          </View>

          {result.gradcam_error ? (
            <Text style={ss.gradcamError}>Grad-CAM error: {result.gradcam_error}</Text>
          ) : null}
        </View>

        {/* ── Image quality ── */}
        <View style={[ss.glassCard, SHADOW_SM]}>
          <View style={ss.cardTitleRow}>
            <ClipboardList size={15} color={COLORS.PRIMARY} strokeWidth={2} />
            <Text style={ss.cardTitle}>Image Quality Check</Text>
          </View>
          <Text style={ss.cardSubtitle}>
            Measured before classification — a failed check blocks the scan.
          </Text>
          <View style={ss.metricRow}>
            <View style={ss.metricBox}>
              <Text style={ss.metricValue}>{quality.blur_score ?? '–'}</Text>
              <Text style={ss.metricLabel}>Sharpness</Text>
              <Text style={ss.metricRange}>min 80</Text>
            </View>
            <View style={ss.metricBox}>
              <Text style={ss.metricValue}>{quality.brightness_score ?? '–'}</Text>
              <Text style={ss.metricLabel}>Brightness</Text>
              <Text style={ss.metricRange}>40 – 215</Text>
            </View>
            <View style={ss.metricBox}>
              <Text
                style={[
                  ss.metricValue,
                  {
                    color: quality.quality_passed ? '#2E7D32' : '#E53935',
                    fontSize: TYPOGRAPHY.LG,
                  },
                ]}
              >
                {quality.quality_passed ? 'PASS' : 'FAIL'}
              </Text>
              <Text style={ss.metricLabel}>Verdict</Text>
              <Text style={ss.metricRange}>
                {quality.issues?.length ? `${quality.issues.length} issue(s)` : 'no issues'}
              </Text>
            </View>
          </View>
        </View>

        {/* ── Recommendation ── */}
        {recommendation && (
          <TouchableOpacity
            style={[ss.treatmentCard, SHADOW_SM]}
            onPress={() => setTreatmentOpen(v => !v)}
            activeOpacity={0.9}
          >
            <View style={ss.treatmentHeaderRow}>
              <View style={ss.treatmentTitleGroup}>
                <View style={ss.treatmentIconBg}>
                  <Zap size={15} color={COLORS.PRIMARY} strokeWidth={2} />
                </View>
                <View>
                  <Text style={ss.treatmentTitle}>Recommended Actions</Text>
                  <Text style={ss.treatmentSeverity}>
                    Severity: {recommendation.severity}
                  </Text>
                </View>
              </View>
              {treatmentOpen ? (
                <ChevronUp size={18} color={COLORS.PRIMARY} strokeWidth={2} />
              ) : (
                <ChevronDown size={18} color={COLORS.PRIMARY} strokeWidth={2} />
              )}
            </View>

            {treatmentOpen && (
              <View style={ss.treatmentBody}>
                {recommendation.warning ? (
                  <Text style={ss.treatmentWarning}>⚠️  {recommendation.warning}</Text>
                ) : null}

                {recommendation.actions?.length ? (
                  <>
                    <Text style={ss.treatmentGroupLabel}>Immediate Actions</Text>
                    {recommendation.actions.map((text, index) => (
                      <View key={text} style={ss.treatmentStep}>
                        <View style={ss.stepNum}>
                          <Text style={ss.stepNumText}>{index + 1}</Text>
                        </View>
                        <Text style={ss.stepText}>{text}</Text>
                      </View>
                    ))}
                  </>
                ) : null}

                {recommendation.prevention?.length ? (
                  <>
                    <Text style={ss.treatmentGroupLabel}>Prevention</Text>
                    {recommendation.prevention.map(text => (
                      <View key={text} style={ss.treatmentStep}>
                        <View style={ss.bullet} />
                        <Text style={ss.stepText}>{text}</Text>
                      </View>
                    ))}
                  </>
                ) : null}

                <Text style={ss.treatmentDisclaimer}>
                  ⚕️  AI screening only — confirm with a licensed veterinarian before
                  treating or medicating birds.
                </Text>
              </View>
            )}
          </TouchableOpacity>
        )}

        {history.length > 0 && renderHistory()}
      </ScrollView>
    );
  };

  // ─── Root render ──────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={ss.safe} edges={['bottom']}>
      <Header notificationCount={0} />
      {phase === PHASE.RESULTS && result ? renderResults() : renderScanner()}
    </SafeAreaView>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const GLASS = {
  backgroundColor: 'rgba(255,255,255,0.82)',
  borderWidth: 1,
  borderColor: 'rgba(255,255,255,0.6)',
  borderRadius: RADIUS.LG,
};

const ss = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#F1F8E9',
  },

  scroll: { flex: 1 },
  scrollContent: { paddingBottom: 36 },
  scannerContent: {
    paddingHorizontal: 16,
    paddingTop: 14,
    paddingBottom: 36,
  },

  // ── Screen header ─────────────────────────────────────────────────────────
  screenHeader: { marginBottom: 10 },
  backBtn: { alignSelf: 'flex-start', marginBottom: 8 },
  backText: {
    color: COLORS.PRIMARY,
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '600',
  },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  screenTitle: {
    fontSize: TYPOGRAPHY.XXL,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
  },
  screenSubtitle: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    marginTop: 2,
  },

  // ── Backend health pill ───────────────────────────────────────────────────
  healthPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderRadius: RADIUS.XL,
    paddingHorizontal: 12,
    paddingVertical: 7,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.06)',
  },
  healthText: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '600',
    color: COLORS.TEXT_SECONDARY,
  },

  // ── Toggle ────────────────────────────────────────────────────────────────
  toggleCard: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderRadius: RADIUS.XL,
    padding: 4,
    borderWidth: 1,
    borderColor: 'rgba(27,94,32,0.15)',
    ...SHADOW_SM,
  },
  toggleBtn: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: RADIUS.LG,
    alignItems: 'center',
  },
  toggleBtnActive: { backgroundColor: COLORS.PRIMARY },
  toggleBtnText: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '600',
    color: COLORS.TEXT_SECONDARY,
  },
  toggleBtnTextActive: { color: '#fff' },
  modeDetects: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    textAlign: 'center',
    marginTop: 7,
    marginBottom: 12,
  },

  // ── Viewfinder ────────────────────────────────────────────────────────────
  vfWrapper: {
    borderRadius: RADIUS.LG,
    overflow: 'hidden',
    ...SHADOW,
  },
  vfBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 14,
    paddingVertical: 9,
    backgroundColor: '#162616',
  },
  liveBadge: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  liveDot: { width: 8, height: 8, borderRadius: 4 },
  liveText: {
    color: '#fff',
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    letterSpacing: 1,
  },
  vfModeLabel: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: TYPOGRAPHY.SM,
  },
  vfFeed: {
    height: VF_H,
    backgroundColor: '#0d1f0d',
    position: 'relative',
    overflow: 'hidden',
  },
  emptyHint: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingHorizontal: 40,
  },
  emptyHintTitle: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
  },
  emptyHintText: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: TYPOGRAPHY.SM,
    textAlign: 'center',
    lineHeight: 17,
  },
  alignTip: {
    position: 'absolute',
    bottom: 10,
    alignSelf: 'center',
    maxWidth: VF_W - 40,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: RADIUS.SM,
    paddingHorizontal: 12,
    paddingVertical: 5,
  },
  alignTipText: {
    color: 'rgba(0,255,65,0.9)',
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '600',
    textAlign: 'center',
  },
  laserLine: {
    position: 'absolute',
    top: G_PAD,
    left: G_PAD,
    right: G_PAD,
    height: 2,
    backgroundColor: '#00FF41',
    shadowColor: '#00FF41',
    shadowOpacity: 0.95,
    shadowRadius: 9,
    shadowOffset: { width: 0, height: 0 },
    elevation: 8,
    zIndex: 10,
  },
  scanStatusOverlay: {
    position: 'absolute',
    bottom: 14,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: RADIUS.SM,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  scanStatusText: {
    color: '#00FF41',
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    letterSpacing: 0.3,
  },

  // ── Source + capture ──────────────────────────────────────────────────────
  sourceRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 12,
  },
  sourceBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
    borderRadius: RADIUS.MD,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderWidth: 1,
    borderColor: 'rgba(27,94,32,0.2)',
  },
  sourceBtnText: {
    color: COLORS.PRIMARY,
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
  },
  btnDisabled: { opacity: 0.45 },

  captureSection: {
    alignItems: 'center',
    paddingTop: 20,
    paddingBottom: 4,
  },
  captureBtn: { alignItems: 'center', gap: 10 },
  captureBtnRing: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 3,
    borderColor: COLORS.PRIMARY,
    alignItems: 'center',
    justifyContent: 'center',
  },
  captureRingDisabled: { borderColor: 'rgba(27,94,32,0.25)' },
  captureBtnCore: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: COLORS.PRIMARY,
    alignItems: 'center',
    justifyContent: 'center',
    ...SHADOW,
  },
  captureCoreDisabled: { backgroundColor: 'rgba(27,94,32,0.3)', elevation: 0 },
  captureBtnLabel: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
    color: COLORS.PRIMARY,
    letterSpacing: 0.3,
  },
  captureLabelDisabled: { color: COLORS.TEXT_SECONDARY },
  processingBanner: { alignItems: 'center', gap: 8 },
  processingTitle: {
    fontSize: TYPOGRAPHY.LG,
    fontWeight: '700',
    color: COLORS.TEXT_PRIMARY,
  },
  processingSubtitle: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    textAlign: 'center',
  },

  // ── Error card ────────────────────────────────────────────────────────────
  errorCard: {
    backgroundColor: '#FFF4F4',
    borderRadius: RADIUS.LG,
    borderWidth: 1,
    borderColor: 'rgba(229,57,53,0.28)',
    padding: 14,
    marginTop: 16,
  },
  errorHeadRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
    marginBottom: 8,
  },
  errorIconBg: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#FFEBEE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorTitle: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '800',
    color: '#C62828',
  },
  errorMessage: {
    fontSize: TYPOGRAPHY.MD,
    color: COLORS.TEXT_SECONDARY,
    lineHeight: 20,
  },
  errorFacts: { marginTop: 10 },
  errorIssue: {
    fontSize: TYPOGRAPHY.SM,
    color: '#C62828',
    fontWeight: '600',
    marginBottom: 3,
  },
  errorMeta: {
    fontSize: TYPOGRAPHY.SM,
    color: '#8a6a6a',
    marginTop: 6,
  },
  errorAction: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    fontStyle: 'italic',
    marginTop: 10,
    lineHeight: 18,
  },
  errorActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    alignSelf: 'flex-start',
    marginTop: 10,
    backgroundColor: '#C62828',
    borderRadius: RADIUS.MD,
    paddingHorizontal: 14,
    paddingVertical: 9,
  },
  errorActionText: {
    color: '#fff',
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
  },

  // ── Results top bar ───────────────────────────────────────────────────────
  resultsTopBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: 16,
    marginTop: 14,
    marginBottom: 10,
  },
  newScanBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: 'rgba(27,94,32,0.1)',
    borderRadius: RADIUS.LG,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  newScanText: {
    color: COLORS.PRIMARY,
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
  },
  resultsTitle: {
    fontSize: TYPOGRAPHY.LG,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
  },
  resultsTimestamp: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
  },

  // ── Diagnosis card ────────────────────────────────────────────────────────
  diseaseCard: {
    marginHorizontal: 16,
    marginBottom: 12,
    borderRadius: RADIUS.LG,
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderWidth: 1,
    padding: 16,
  },
  diseaseCardTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  diseaseCardLeft: { flex: 1, paddingRight: 10 },
  diagnosisLabel: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginBottom: 4,
  },
  diseaseName: {
    fontSize: TYPOGRAPHY.XXL,
    fontWeight: '900',
    letterSpacing: -0.3,
  },
  diseaseVirusLabel: {
    fontSize: TYPOGRAPHY.SM,
    color: '#888',
    marginTop: 2,
    fontStyle: 'italic',
  },
  diseaseIconBg: {
    width: 54,
    height: 54,
    borderRadius: 27,
    alignItems: 'center',
    justifyContent: 'center',
  },
  confidenceHeadline: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
    marginBottom: 12,
  },
  confidenceValue: {
    fontSize: TYPOGRAPHY.XXXL,
    fontWeight: '900',
    letterSpacing: -1,
  },
  confidenceCaption: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
  },
  diseaseTagRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    alignItems: 'center',
  },
  riskBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    borderRadius: RADIUS.XL,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  riskBadgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  diseaseTag: {
    backgroundColor: 'rgba(0,0,0,0.06)',
    borderRadius: RADIUS.LG,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  diseaseTagText: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: '600',
  },

  // ── Glass card (shared) ───────────────────────────────────────────────────
  glassCard: {
    ...GLASS,
    marginHorizontal: 16,
    marginBottom: 12,
    padding: 14,
  },
  cardTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    marginBottom: 4,
  },
  cardTitle: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
    color: COLORS.TEXT_PRIMARY,
  },
  cardSubtitle: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    marginBottom: 12,
    lineHeight: 18,
  },

  // ── Probability bars ──────────────────────────────────────────────────────
  confRow: { marginVertical: 5 },
  confMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  confLabel: {
    fontSize: TYPOGRAPHY.MD,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: '500',
  },
  confLabelStrong: {
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
  },
  confPct: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
  },
  confTrack: {
    height: 8,
    backgroundColor: 'rgba(0,0,0,0.08)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  confFill: { height: '100%', borderRadius: 4 },
  confDivider: {
    height: 1,
    backgroundColor: 'rgba(0,0,0,0.06)',
    marginVertical: 4,
  },
  modelTag: {
    fontSize: 10,
    color: '#aaa',
    marginTop: 10,
    textAlign: 'right',
    fontStyle: 'italic',
  },

  // ── Sample vs Grad-CAM ────────────────────────────────────────────────────
  compRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  compCol: { alignItems: 'center', gap: 6 },
  compPhoto: {
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: '#1a1f1a',
  },
  compEmpty: { alignItems: 'center', justifyContent: 'center' },
  compCaption: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: '600',
  },
  remoteOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
    backgroundColor: 'rgba(26,31,26,0.75)',
  },
  remoteFailText: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 10,
  },
  gradcamError: {
    fontSize: TYPOGRAPHY.SM,
    color: '#C62828',
    marginTop: 8,
  },

  // ── Quality metrics ───────────────────────────────────────────────────────
  metricRow: { flexDirection: 'row', gap: 10 },
  metricBox: {
    flex: 1,
    backgroundColor: 'rgba(27,94,32,0.06)',
    borderRadius: RADIUS.MD,
    paddingVertical: 12,
    alignItems: 'center',
    gap: 2,
  },
  metricValue: {
    fontSize: TYPOGRAPHY.XL,
    fontWeight: '800',
    color: COLORS.TEXT_PRIMARY,
  },
  metricLabel: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    fontWeight: '600',
  },
  metricRange: { fontSize: 10, color: '#aaa' },

  // ── History ───────────────────────────────────────────────────────────────
  historyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,0,0,0.06)',
  },
  historyThumb: {
    width: 40,
    height: 40,
    borderRadius: 8,
    backgroundColor: '#1a1f1a',
  },
  historyBody: { flex: 1 },
  historyName: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
  },
  historyMeta: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    marginTop: 1,
  },
  historyDot: { width: 10, height: 10, borderRadius: 5 },

  // ── Recommendation card ───────────────────────────────────────────────────
  treatmentCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderRadius: RADIUS.LG,
    borderWidth: 1,
    borderColor: 'rgba(27,94,32,0.15)',
    overflow: 'hidden',
  },
  treatmentHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 14,
  },
  treatmentTitleGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
  },
  treatmentIconBg: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#E8F5E9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  treatmentTitle: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
    color: COLORS.TEXT_PRIMARY,
  },
  treatmentSeverity: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    marginTop: 1,
  },
  treatmentBody: {
    paddingHorizontal: 14,
    paddingBottom: 14,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,0,0,0.06)',
  },
  treatmentWarning: {
    fontSize: TYPOGRAPHY.MD,
    color: '#C62828',
    fontWeight: '600',
    marginTop: 12,
    lineHeight: 20,
  },
  treatmentGroupLabel: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    color: COLORS.TEXT_SECONDARY,
    textTransform: 'uppercase',
    letterSpacing: 0.7,
    marginTop: 14,
    marginBottom: 8,
  },
  treatmentDisclaimer: {
    fontSize: TYPOGRAPHY.SM,
    color: '#888',
    fontStyle: 'italic',
    marginTop: 14,
    lineHeight: 18,
  },
  treatmentStep: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 10,
  },
  stepNum: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: COLORS.PRIMARY,
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    marginTop: 1,
  },
  stepNumText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '800',
  },
  bullet: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.PRIMARY,
    marginTop: 8,
    marginLeft: 8,
    marginRight: 8,
    flexShrink: 0,
  },
  stepText: {
    flex: 1,
    fontSize: TYPOGRAPHY.MD,
    color: COLORS.TEXT_SECONDARY,
    lineHeight: 20,
  },
});
