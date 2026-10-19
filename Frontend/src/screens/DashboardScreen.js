import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  RefreshControl,
} from 'react-native';
import { Camera, Bug, Shield } from 'lucide-react-native';
import Header from '../components/Header';
import ChickenBehaviourModule from './ChickenBehaviourModule';
import { COLORS, GLASS_STYLE, SHADOW, SHADOW_SM, TYPOGRAPHY, RADIUS } from '../../theme';
import { fetchChickenStatus, fetchChickenEvents } from '../services/chickenApi';
import { checkDiseaseApiHealth } from '../services/diseaseApi';
import { fetchPredatorSummary } from '../services/predatorApi';

const MODULES = [
  {
    key: 'ChickenBehaviour',
    label: 'Chicken Behaviour',
    icon: Camera,
    color: COLORS.PRIMARY,
    bg: '#E8F5E9',
    active: true,
  },
  {
    key: 'DiseaseDetection',
    label: 'Disease Detection',
    icon: Bug,
    color: '#E53935',
    bg: '#FFEBEE',
    active: true,
  },
  {
    key: 'PredatorDetection',
    label: 'Predator Alert',
    icon: Shield,
    color: '#5C35D3',
    bg: '#EDE7F6',
    active: true,
  },
];

const STATS_POLL_MS = 15000;

// Shown while the first round of requests is still in flight.
const LOADING_STATS = [
  { label: 'Alerts Today', value: '…', unit: 'events', color: COLORS.DANGER },
  { label: 'System Health', value: '…', unit: 'services up', color: COLORS.PRIMARY_LIGHT },
];

/**
 * The quick-stats bar, built from what the three detectors actually report:
 *
 *   Alerts Today   chicken  GET /api/events        -> ABNORMAL checks today
 *                 + predator GET /api/stats/summary -> today.total
 *   System Health   how many of the three services answered
 *
 * Every request is settled independently, so one dead service degrades a
 * single number instead of blanking the whole bar.
 */
async function loadFarmStats() {
  const [chickenStatus, chickenEvents, predatorSummary, diseaseHealth] =
    await Promise.allSettled([
      fetchChickenStatus(),
      fetchChickenEvents(100),
      fetchPredatorSummary(),
      checkDiseaseApiHealth(),
    ]);

  const chickenUp = chickenStatus.status === 'fulfilled';
  const predatorUp = predatorSummary.status === 'fulfilled';
  const diseaseUp = diseaseHealth.status === 'fulfilled';
  const servicesUp = [chickenUp, predatorUp, diseaseUp].filter(Boolean).length;

  const today = new Date().toDateString();
  const abnormalToday =
    chickenEvents.status === 'fulfilled'
      ? (chickenEvents.value.events || []).filter(event => {
          if (event.result !== 'ABNORMAL') return false;
          const at = new Date(event.timestamp);
          return !Number.isNaN(at.getTime()) && at.toDateString() === today;
        }).length
      : 0;

  const predatorToday = predatorUp ? predatorSummary.value.today.total : 0;
  const alertsToday = abnormalToday + predatorToday;

  return [
    {
      label: 'Alerts Today',
      value: chickenUp || predatorUp ? String(alertsToday) : '—',
      unit: 'events',
      color: COLORS.DANGER,
    },
    {
      label: 'System Health',
      value: `${servicesUp}/3`,
      unit: 'services up',
      color: servicesUp === 3 ? COLORS.PRIMARY_LIGHT : COLORS.WARNING,
    },
  ];
}

export default function DashboardScreen({ navigation }) {
  const [stats, setStats] = useState(LOADING_STATS);
  const [alertCount, setAlertCount] = useState(0);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    const next = await loadFarmStats();
    setStats(next);
    const alerts = Number(next[0].value);
    setAlertCount(Number.isNaN(alerts) ? 0 : alerts);
  }, []);

  useEffect(() => {
    let active = true;
    const run = () => {
      load().catch(() => {
        // loadFarmStats settles every request itself; this only guards against
        // an unexpected throw so the interval keeps running.
      });
    };
    run();
    const timer = setInterval(() => {
      if (active) run();
    }, STATS_POLL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [load]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await load().catch(() => {});
    setRefreshing(false);
  }, [load]);

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <Header notificationCount={alertCount} />

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={COLORS.PRIMARY} />
        }
      >
        {/* Quick Stats Bar */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          nestedScrollEnabled
          contentContainerStyle={styles.statsRow}
          style={styles.statsScroll}
        >
          {stats.map(stat => (
            <View key={stat.label} style={[styles.statCard, SHADOW_SM]}>
              <Text style={[styles.statValue, { color: stat.color }]}>{stat.value}</Text>
              <Text style={styles.statLabel}>{stat.label}</Text>
              <Text style={styles.statUnit}>{stat.unit}</Text>
            </View>
          ))}
        </ScrollView>

        {/* Module Navigation Cards */}
        <Text style={styles.sectionLabel}>AI Monitoring Modules</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          nestedScrollEnabled
          contentContainerStyle={styles.modulesRow}
          style={styles.modulesScroll}
        >
          {MODULES.map(mod => {
            const Icon = mod.icon;
            return (
              <TouchableOpacity
                key={mod.key}
                style={[styles.moduleCard, SHADOW, { backgroundColor: COLORS.WHITE }]}
                onPress={() => navigation.navigate(mod.key)}
                activeOpacity={0.8}
              >
                <View style={[styles.moduleIconBg, { backgroundColor: mod.bg }]}>
                  <Icon size={26} color={mod.color} strokeWidth={1.8} />
                </View>
                <Text style={styles.moduleLabel}>{mod.label}</Text>
                <View style={[styles.activeIndicator, { backgroundColor: mod.active ? COLORS.PRIMARY_LIGHT : '#ddd' }]} />
              </TouchableOpacity>
            );
          })}
        </ScrollView>

        {/* Divider */}
        <View style={styles.divider}>
          <View style={styles.dividerLine} />
          <Text style={styles.dividerLabel}>Live Detection Feed</Text>
          <View style={styles.dividerLine} />
        </View>

        {/* Chicken Behaviour Module (default view) */}
        <ChickenBehaviourModule />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.BACKGROUND,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 32,
  },
  statsScroll: {
    marginTop: 16,
  },
  statsRow: {
    flexGrow: 1,
    paddingHorizontal: 16,
    gap: 10,
  },
  statCard: {
    backgroundColor: COLORS.WHITE,
    borderRadius: RADIUS.MD,
    paddingVertical: 12,
    paddingHorizontal: 16,
    alignItems: 'center',
    flex: 1,
    minWidth: 100,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.05)',
  },
  statValue: {
    fontSize: TYPOGRAPHY.XL,
    fontWeight: '800',
  },
  statLabel: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.TEXT_SECONDARY,
    marginTop: 2,
    fontWeight: '500',
  },
  statUnit: {
    fontSize: 10,
    color: COLORS.TEXT_SECONDARY,
    marginTop: 1,
  },
  sectionLabel: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '700',
    color: COLORS.TEXT_SECONDARY,
    textTransform: 'uppercase',
    letterSpacing: 0.7,
    marginHorizontal: 16,
    marginTop: 20,
    marginBottom: 10,
  },
  modulesScroll: {},
  modulesRow: {
    paddingHorizontal: 16,
    gap: 12,
  },
  moduleCard: {
    width: 130,
    borderRadius: RADIUS.LG,
    padding: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.05)',
  },
  moduleIconBg: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  moduleLabel: {
    fontSize: TYPOGRAPHY.SM,
    fontWeight: '600',
    color: COLORS.TEXT_PRIMARY,
    textAlign: 'center',
    lineHeight: 16,
  },
  activeIndicator: {
    width: 24,
    height: 4,
    borderRadius: 2,
    marginTop: 8,
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 16,
    marginTop: 22,
    marginBottom: 4,
    gap: 10,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: 'rgba(27,94,32,0.15)',
  },
  dividerLabel: {
    fontSize: TYPOGRAPHY.SM,
    color: COLORS.PRIMARY,
    fontWeight: '600',
    letterSpacing: 0.3,
  },
});
