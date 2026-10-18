import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  Image,
  Modal,
  Animated,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { WebView } from 'react-native-webview';
import { Siren, Volume2, VolumeX, X } from 'lucide-react-native';
import { COLORS, RADIUS, TYPOGRAPHY } from '../../theme';
import { speciesEmoji, speciesLabel } from '../utils/predatorAlarmState';

// Palette of the (dark) predator screen this card sits on top of.
const DARK = {
  CARD: '#141414',
  CARD_BORDER: 'rgba(255,90,70,0.45)',
  TEXT: '#F0F0F0',
  TEXT_SECONDARY: '#8A8A8A',
  IMAGE_BG: '#000',
};

const DANGER_RED = '#C1121F';
const DANGER_RED_DEEP = '#7A0B14';

// The siren is generated with the Web Audio API inside a 1px WebView, the same
// way the chicken module's alarm is: react-native-webview is already a
// dependency of this screen (it carries the MJPEG feed), so the alarm needs no
// audio library and no native rebuild - it ships with a JS reload.
// startSiren()/stopSiren() are driven from RN by injectJavaScript, and
// mediaPlaybackRequiresUserAction={false} is what lets it sound without a tap.
//
// The tone is deliberately not the chicken alarm's two beeps: this one is a
// rising/falling sawtooth wail, so a predator alert is audibly different from a
// behaviour alert when both backends are running.
const SIREN_HTML = `<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;background:transparent">
<script>
  var ctx = null, timer = null;

  function audio() {
    if (!ctx) { ctx = new (window.AudioContext || window.webkitAudioContext)(); }
    if (ctx.state === 'suspended') { ctx.resume(); }
    return ctx;
  }

  // One swept note. Ramped rather than switched, so it does not click.
  function wail(from, to, at, dur) {
    var c = audio();
    var osc = c.createOscillator();
    var gain = c.createGain();
    var t0 = c.currentTime + at;
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(from, t0);
    osc.frequency.linearRampToValueAtTime(to, t0 + dur);
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(0.35, t0 + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(gain);
    gain.connect(c.destination);
    osc.start(t0);
    osc.stop(t0 + dur + 0.03);
  }

  // Up-down emergency wail, repeated until stopped.
  function cycle() { wail(620, 1180, 0, 0.34); wail(1180, 620, 0.36, 0.34); }

  function startSiren() {
    if (timer) { return; }
    cycle();
    timer = setInterval(cycle, 760);
  }

  function stopSiren() {
    if (timer) { clearInterval(timer); timer = null; }
  }
</script></body></html>`;

/** Detection time, to the second - an alarm is about a moment, not a day. */
function formatAlarmTime(iso) {
  if (!iso) return '-';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  const time = date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
  return date.toDateString() === new Date().toDateString()
    ? `Today, ${time}`
    : `${date.toLocaleDateString()}, ${time}`;
}

function Fact({ label, value, color, emphasis }) {
  return (
    <View style={styles.factRow}>
      <Text style={styles.factLabel}>{label}</Text>
      <Text
        style={[
          styles.factValue,
          color ? { color } : null,
          emphasis ? styles.factValueStrong : null,
        ]}
        numberOfLines={1}
      >
        {value}
      </Text>
    </View>
  );
}

/**
 * The predator danger alarm: a siren plus a red emergency popup.
 *
 * Presentational on purpose - it sounds whenever `alarm` is set and `muted` is
 * false, and the screen owns the deciding. Nothing here talks to the API.
 *
 * @param {?object} alarm     alarm details from predatorAlarmState, or null when idle
 * @param {?string} imageUri  resolved URL/URI of the detection frame
 * @param {boolean} muted     session mute - silences the siren, the card stays up
 * @param {Function} onStop        stop the siren and close the card
 * @param {Function} onToggleMute  flip the session mute
 */
export default function PredatorAlarm({
  alarm,
  imageUri,
  muted,
  onStop,
  onToggleMute,
  imageLoading = false,
}) {
  const sirenRef = useRef(null);
  const sirenReadyRef = useRef(false);
  const pulse = useRef(new Animated.Value(1)).current;

  const visible = Boolean(alarm);
  const sounding = visible && !muted;

  // -- siren ------------------------------------------------------------------
  useEffect(() => {
    if (!sirenReadyRef.current) return;
    sirenRef.current?.injectJavaScript(`${sounding ? 'startSiren' : 'stopSiren'}(); true;`);
  }, [sounding]);

  // Leaving the screen must never leave the siren running.
  useEffect(
    () => () => {
      sirenRef.current?.injectJavaScript('stopSiren(); true;');
    },
    []
  );

  // -- card pulse -------------------------------------------------------------
  useEffect(() => {
    if (!visible) {
      pulse.setValue(1);
      return undefined;
    }
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 0.35, duration: 460, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 1, duration: 460, useNativeDriver: true }),
      ])
    );
    anim.start();
    return () => anim.stop();
  }, [visible, pulse]);

  const confidence =
    alarm && typeof alarm.confidence === 'number'
      ? `${Math.round(alarm.confidence * 100)}%`
      : '-';

  return (
    <>
      {/* The siren WebView must stay laid out (a display:none view can be
          dropped by the platform), so it is parked off-screen at 1x1. */}
      <View style={styles.sirenWrap} pointerEvents="none">
        <WebView
          ref={sirenRef}
          source={{ html: SIREN_HTML }}
          originWhitelist={['*']}
          javaScriptEnabled
          mediaPlaybackRequiresUserAction={false}
          allowsInlineMediaPlayback
          style={styles.siren}
          onLoadEnd={() => {
            sirenReadyRef.current = true;
            if (sounding) sirenRef.current?.injectJavaScript('startSiren(); true;');
          }}
        />
      </View>

      <Modal visible={visible} transparent animationType="fade" onRequestClose={onStop}>
        <View style={styles.backdrop}>
          <View style={styles.card}>
            {/* Header */}
            <View style={styles.header}>
              <Animated.View style={{ opacity: pulse }}>
                <Siren size={22} color={COLORS.WHITE} strokeWidth={2.4} />
              </Animated.View>
              <Text style={styles.headerText} numberOfLines={1}>
                🚨 PREDATOR DETECTED
              </Text>
              <TouchableOpacity onPress={onStop} hitSlop={10} activeOpacity={0.7}>
                <X size={20} color="rgba(255,255,255,0.85)" strokeWidth={2.4} />
              </TouchableOpacity>
            </View>

            {/* Captured frame */}
            <View style={styles.imageBox}>
              {imageUri ? (
                <Image source={{ uri: imageUri }} style={styles.image} resizeMode="contain" />
              ) : (
                <View style={styles.imageEmpty}>
                  {imageLoading ? (
                    <ActivityIndicator size="small" color={DANGER_RED} />
                  ) : (
                    <Text style={styles.imageEmptyIcon}>{speciesEmoji(alarm?.species)}</Text>
                  )}
                  <Text style={styles.imageEmptyText}>
                    {imageLoading
                      ? 'Fetching the captured frame...'
                      : 'No frame saved for this alert.'}
                  </Text>
                </View>
              )}
            </View>

            {/* Facts */}
            <View style={styles.facts}>
              <Fact
                label="ANIMAL"
                value={`${speciesEmoji(alarm?.species)}  ${speciesLabel(alarm?.species)}`}
                emphasis
              />
              <Fact label="CONFIDENCE" value={confidence} color={DANGER_RED} emphasis />
              <Fact label="TIME" value={formatAlarmTime(alarm?.at)} />
              <View style={styles.factRow}>
                <Text style={styles.factLabel}>STATUS</Text>
                <View style={styles.statusPill}>
                  <Animated.View style={[styles.statusDot, { opacity: pulse }]} />
                  <Text style={styles.statusText}>DANGER</Text>
                </View>
              </View>
            </View>

            <Text style={styles.source} numberOfLines={2}>
              {alarm?.message}
              {alarm?.source ? `  ·  source: ${alarm.source}` : ''}
            </Text>

            {/* Controls */}
            <View style={styles.buttons}>
              <TouchableOpacity
                style={[styles.btn, styles.btnStop]}
                onPress={onStop}
                activeOpacity={0.85}
              >
                <X size={16} color={COLORS.WHITE} strokeWidth={2.4} />
                <Text style={styles.btnText}>Stop Alarm</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.btn, styles.btnMute, muted && styles.btnMuted]}
                onPress={onToggleMute}
                activeOpacity={0.85}
              >
                {muted ? (
                  <VolumeX size={16} color={COLORS.WHITE} strokeWidth={2.4} />
                ) : (
                  <Volume2 size={16} color={COLORS.WHITE} strokeWidth={2.4} />
                )}
                <Text style={styles.btnText}>{muted ? 'Unmute' : 'Mute'}</Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.footnote}>
              {muted
                ? 'Sound is off for this session - the card still appears on every detection.'
                : 'One confirmed snake or scorpion is enough to raise this alarm.'}
            </Text>
          </View>
        </View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  sirenWrap: {
    position: 'absolute',
    width: 1,
    height: 1,
    top: -10,
    left: -10,
    opacity: 0,
  },
  siren: {
    width: 1,
    height: 1,
    backgroundColor: 'transparent',
  },
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.78)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  card: {
    width: '100%',
    maxWidth: 420,
    backgroundColor: DARK.CARD,
    borderRadius: RADIUS.LG,
    borderWidth: 2,
    borderColor: DARK.CARD_BORDER,
    overflow: 'hidden',
    paddingBottom: 14,
    elevation: 14,
    shadowColor: DANGER_RED,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.5,
    shadowRadius: 14,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: DANGER_RED,
    paddingHorizontal: 14,
    paddingVertical: 13,
  },
  headerText: {
    flex: 1,
    color: COLORS.WHITE,
    fontSize: TYPOGRAPHY.LG,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  imageBox: {
    height: 190,
    backgroundColor: DARK.IMAGE_BG,
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  imageEmpty: {
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 20,
  },
  imageEmptyIcon: {
    fontSize: 34,
  },
  imageEmptyText: {
    color: DARK.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.SM,
    textAlign: 'center',
  },
  facts: {
    paddingHorizontal: 14,
    paddingTop: 12,
  },
  factRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 7,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.07)',
    gap: 12,
  },
  factLabel: {
    fontSize: 10,
    color: DARK.TEXT_SECONDARY,
    fontWeight: '700',
    letterSpacing: 0.9,
  },
  factValue: {
    flexShrink: 1,
    fontSize: TYPOGRAPHY.MD,
    color: DARK.TEXT,
    fontWeight: '600',
    textAlign: 'right',
  },
  factValueStrong: {
    fontSize: TYPOGRAPHY.XL,
    fontWeight: '800',
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    backgroundColor: DANGER_RED_DEEP,
    borderRadius: RADIUS.SM,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.WHITE,
  },
  statusText: {
    color: COLORS.WHITE,
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  source: {
    fontSize: TYPOGRAPHY.SM,
    color: DARK.TEXT_SECONDARY,
    paddingHorizontal: 14,
    marginTop: 10,
    lineHeight: 16,
  },
  buttons: {
    flexDirection: 'row',
    gap: 10,
    paddingHorizontal: 14,
    marginTop: 14,
  },
  btn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    borderRadius: RADIUS.MD,
    paddingVertical: 12,
  },
  btnStop: {
    backgroundColor: DANGER_RED,
  },
  btnMute: {
    backgroundColor: 'rgba(255,255,255,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.16)',
  },
  btnMuted: {
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
  btnText: {
    fontSize: TYPOGRAPHY.MD,
    fontWeight: '700',
    color: COLORS.WHITE,
  },
  footnote: {
    fontSize: TYPOGRAPHY.SM,
    color: DARK.TEXT_SECONDARY,
    textAlign: 'center',
    paddingHorizontal: 14,
    marginTop: 10,
    lineHeight: 15,
  },
});
