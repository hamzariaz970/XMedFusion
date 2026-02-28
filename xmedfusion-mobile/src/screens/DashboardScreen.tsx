import React from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  StatusBar, ScrollView,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { 
  Scan, 
  Activity, 
  TrendingUp, 
  BarChart3, 
  Clock, 
  ChevronRight,
  ShieldAlert,
  ShieldCheck,
  Zap,
  ClipboardList,
  Users
} from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

const stats = [
  { label: 'Reports Today', value: '12', trend: '+3', icon: BarChart3 },
  { label: 'Avg Confidence', value: '94%', trend: '+2%', icon: TrendingUp },
  { label: 'Pending Review', value: '2', trend: '', icon: Clock },
];

const recentCases = [
  { id: 'C001', patient: 'PT-2847', finding: 'Bilateral Consolidation', severity: 'critical', time: '2h ago' },
  { id: 'C002', patient: 'PT-2846', finding: 'Cardiomegaly Detected', severity: 'warning', time: '4h ago' },
  { id: 'C003', patient: 'PT-2845', finding: 'Normal Radiograph', severity: 'normal', time: '6h ago' },
];

export default function DashboardScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const s = styles(theme);

  const severityColor = (sev: string) =>
    sev === 'critical' ? theme.destructive : sev === 'warning' ? theme.warning : theme.success;
  const severityBg = (sev: string) =>
    sev === 'critical' ? theme.destructiveBg : sev === 'warning' ? theme.warningBg : theme.successBg;

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <View style={s.header}>
        <View>
          <Text style={s.greeting}>Good Morning,</Text>
          <Text style={s.doctorName}>Dr. Ahmad  👋</Text>
        </View>
        <TouchableOpacity style={s.avatarBtn} onPress={() => navigation.navigate('Profile')}>
          <Text style={s.avatarText}>DA</Text>
        </TouchableOpacity>
      </View>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={s.scroll}>
        {/* Stats */}
        <View style={s.statsRow}>
          {stats.map((stat) => (
            <View key={stat.label} style={s.statCard}>
              <stat.icon size={20} color={theme.primary} strokeWidth={2} style={{ marginBottom: 4 }} />
              <Text style={s.statValue}>{stat.value}</Text>
              {stat.trend ? <Text style={s.statTrend}>{stat.trend}</Text> : null}
              <Text style={s.statLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>

        {/* Primary CTA */}
        <TouchableOpacity style={s.primaryAction} onPress={() => navigation.navigate('Upload')} activeOpacity={0.85}>
          <View style={s.primaryActionContent}>
            <View style={s.primaryCircle}>
               <Scan color={theme.primaryForeground} size={28} strokeWidth={2.5} />
            </View>
            <View>
              <Text style={s.primaryActionTitle}>Upload New X-Ray</Text>
              <Text style={s.primaryActionSubtitle}>AI-powered analysis in ~30 seconds</Text>
            </View>
          </View>
          <ChevronRight color={theme.primaryForeground} size={24} strokeWidth={3} />
        </TouchableOpacity>

        {/* Quick Actions */}
        <Text style={s.sectionTitle}>Quick Access</Text>
        <View style={s.quickActionsGrid}>
          {[
            { label: 'Analyze Scan', icon: Zap, desc: 'Process new radiograph', color: theme.primary, target: 'Upload' },
            { label: 'Case Library', icon: ClipboardList, desc: 'View previous reports', color: theme.accent, target: 'Patients' },
          ].map((action) => (
            <TouchableOpacity 
              key={action.label} 
              style={s.quickAction} 
              activeOpacity={0.8}
              onPress={() => navigation.navigate(action.target)}
            >
              <View style={[s.actionIcon, { backgroundColor: action.color + '20' }]}>
                <action.icon color={action.color} size={24} strokeWidth={2.5} />
              </View>
              <Text style={s.quickActionLabel}>{action.label}</Text>
              <Text style={s.quickActionDesc}>{action.desc}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Recent Cases */}
        <View style={s.sectionRow}>
          <Text style={s.sectionTitle}>Recent Cases</Text>
          <TouchableOpacity onPress={() => navigation.navigate('Patients')}>
            <Text style={s.seeAll}>See All →</Text>
          </TouchableOpacity>
        </View>
        <View style={s.caseList}>
          {recentCases.map((c) => (
            <TouchableOpacity 
              key={c.id} 
              style={s.caseCard} 
              activeOpacity={0.8}
              onPress={() => navigation.navigate('ReportDetail', { report: c })}
            >
              <View style={[s.caseIcon, { backgroundColor: severityBg(c.severity) }]}>
                {c.severity === 'critical' ? (
                  <ShieldAlert color={theme.destructive} size={20} />
                ) : c.severity === 'warning' ? (
                  <Activity color={theme.warning} size={20} />
                ) : (
                  <ShieldCheck color={theme.success} size={20} />
                )}
              </View>
              <View style={s.caseInfo}>
                <Text style={s.casePatient}>{c.patient}</Text>
                <Text style={s.caseFinding}>{c.finding}</Text>
              </View>
              <View style={s.caseMeta}>
                <View style={[s.severityBadge, { backgroundColor: severityBg(c.severity) }]}>
                  <Text style={[s.severityText, { color: severityColor(c.severity) }]}>{c.severity}</Text>
                </View>
                <Text style={s.caseTime}>{c.time}</Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.background },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: spacing.lg, paddingTop: spacing.xxl, paddingBottom: spacing.md },
  greeting: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.regular },
  doctorName: { color: theme.foreground, fontSize: typography.xl, fontWeight: '700', fontFamily: fontFamily.bold },
  avatarBtn: { width: 42, height: 42, borderRadius: 21, backgroundColor: theme.primaryGlow, borderWidth: 2, borderColor: theme.primary, alignItems: 'center', justifyContent: 'center' },
  avatarText: { color: theme.primary, fontWeight: '700', fontSize: typography.sm, fontFamily: fontFamily.bold },
  scroll: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl },
  statsRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.lg },
  statCard: { flex: 1, backgroundColor: theme.card, borderRadius: radius.md, borderWidth: 1, borderColor: theme.cardBorder, padding: spacing.md, alignItems: 'center' },
  statValue: { color: theme.primary, fontSize: typography['2xl'], fontWeight: '700', fontFamily: fontFamily.bold },
  statTrend: { color: theme.success, fontSize: typography.xs, fontFamily: fontFamily.medium },
  statLabel: { color: theme.mutedForeground, fontSize: typography.xs, textAlign: 'center', marginTop: 2, fontFamily: fontFamily.regular },
  primaryAction: { backgroundColor: theme.primary, borderRadius: radius.lg, padding: spacing.lg, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.lg, shadowColor: theme.primary, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.4, shadowRadius: 12, elevation: 8 },
  primaryActionContent: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  primaryCircle: { width: 48, height: 48, borderRadius: 24, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  primaryActionTitle: { color: theme.primaryForeground, fontWeight: '700', fontSize: typography.lg, fontFamily: fontFamily.bold },
  primaryActionSubtitle: { color: theme.primaryForeground, fontSize: typography.xs, marginTop: 2, fontFamily: fontFamily.regular, opacity: 0.7 },
  sectionRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.md },
  sectionTitle: { color: theme.foreground, fontSize: typography.lg, fontWeight: '600', marginBottom: spacing.md, fontFamily: fontFamily.semiBold },
  seeAll: { color: theme.primary, fontSize: typography.sm, fontFamily: fontFamily.medium },
  quickActionsGrid: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.lg },
  quickAction: { flex: 1, backgroundColor: theme.card, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.lg, padding: spacing.md },
  actionIcon: { width: 44, height: 44, borderRadius: 12, alignItems: 'center', justifyContent: 'center', marginBottom: spacing.sm },
  quickActionLabel: { color: theme.foreground, fontWeight: '600', fontSize: typography.sm, marginBottom: 2, fontFamily: fontFamily.semiBold },
  quickActionDesc: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  caseList: { gap: spacing.sm },
  caseCard: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.card, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.md, padding: spacing.md, gap: spacing.md },
  caseIcon: { width: 44, height: 44, borderRadius: radius.sm, alignItems: 'center', justifyContent: 'center' },
  caseInfo: { flex: 1 },
  casePatient: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  caseFinding: { color: theme.foreground, fontSize: typography.sm, fontWeight: '500', marginTop: 2, fontFamily: fontFamily.medium },
  caseMeta: { alignItems: 'flex-end', gap: 4 },
  severityBadge: { paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: radius.full },
  severityText: { fontSize: typography.xs, fontWeight: '600', textTransform: 'capitalize', fontFamily: fontFamily.semiBold },
  caseTime: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
});
