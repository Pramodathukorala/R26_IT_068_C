import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  Image,
  Animated,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Bell } from 'lucide-react-native';
import { COLORS, SHADOW_SM, TYPOGRAPHY } from '../../theme';

export default function Header({ notificationCount = 0 }) {
  const insets = useSafeAreaInsets();
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 0.2,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, [pulseAnim]);

  return (
    <View style={[styles.container, { paddingTop: insets.top + 10 }]}>
      {/* Logo */}
      <Image
        source={require('../../assets/logo.png')}
        style={styles.logo}
        resizeMode="contain"
      />

      {/* System Status */}
      <View style={styles.statusCenter}>
        <Animated.View style={[styles.dot, { opacity: pulseAnim }]} />
        <Text style={styles.statusText}>Monitoring Active</Text>
      </View>

      {/* Bell Icon */}
      <TouchableOpacity style={styles.bellWrapper} activeOpacity={0.7}>
        <Bell size={22} color={COLORS.PRIMARY} strokeWidth={2} />
        {notificationCount > 0 && (
          <View style={styles.badge}>
            <Text style={styles.badgeText}>
              {notificationCount > 9 ? '9+' : notificationCount}
            </Text>
          </View>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingBottom: 12,
    backgroundColor: COLORS.WHITE,
    ...SHADOW_SM,
  },
  logo: {
    width: 120,
    height: 38,
  },
  statusCenter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  dot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    backgroundColor: COLORS.PRIMARY_LIGHT,
  },
  statusText: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.PRIMARY,
    fontWeight: '600',
    letterSpacing: 0.3,
  },
  bellWrapper: {
    padding: 6,
    position: 'relative',
  },
  badge: {
    position: 'absolute',
    top: 2,
    right: 2,
    backgroundColor: COLORS.DANGER,
    borderRadius: 8,
    minWidth: 16,
    height: 16,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 3,
  },
  badgeText: {
    color: COLORS.WHITE,
    fontSize: 9,
    fontWeight: '700',
  },
});
