import React, { useEffect, useMemo, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, ScrollView, ActivityIndicator } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Activity, AlertTriangle, ArrowRight, FileText, ImagePlus, Search, ShieldCheck, UserRound, Users } from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { createShadow, glassPanel, mutedCard, shellBackground, surfaceCard } from '../theme/ui';
import { useAuth } from '../theme/AuthContext';
import { Patient, usePatientContext } from '../context/PatientContext';
import { supabase } from '../lib/supabase';

interface DoctorProfile {
  full_name: string;
  specialization: string;
}

interface RecentScan {
  id: string;
  patient_id: string;
  created_at: string;
  scan_type: string | null;
  severity: string | null;
  impression: string | null;
}

interface HilTaskSummary {
  id: string;
  title: string;
  total_scans: number;
  completed_scans: number;
  status: string;
}

const getGreeting = () => {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good Morning';
  if (hour < 17) return 'Good Afternoon';
  return 'Good Evening';
};

const startOfWeek = () => {
  const date = new Date();
  const day = date.getDay();
  date.setDate(date.getDate() - day);
  date.setHours(0, 0, 0, 0);
  return date;
};

export default function DashboardScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const { user, signOut } = useAuth();
  const { patients, selectedPatient, setSelectedPatient, loading: patientsLoading } = usePatientContext();
  const [doctorProfile, setDoctorProfile] = useState<DoctorProfile | null>(null);
  const [recentScans, setRecentScans] = useState<RecentScan[]>([]);
  const [hilTasks, setHilTasks] = useState<HilTaskSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const s = styles(theme);

  useEffect(() => {
    if (!user) return;

    const loadDashboard = async () => {
      setLoading(true);
      try {
        const [{ data: doctor }, { data: tasks }] = await Promise.all([
          supabase
            .from('doctors')
            .select('full_name, specialization')
            .eq('user_id', user.id)
            .maybeSingle(),
          supabase
            .from('hil_tasks')
            .select('id, title, total_scans, completed_scans, status')
            .eq('doctor_id', user.id)
            .in('status', ['assigned', 'in_progress'])
            .order('created_at', { ascending: false })
            .limit(4),
        ]);

        if (doctor) setDoctorProfile(doctor as DoctorProfile);
        setHilTasks((tasks || []) as HilTaskSummary[]);
      } finally {
        setLoading(false);
      }
    };

    void loadDashboard();
  }, [user]);

  useEffect(() => {
    const loadRecentScans = async () => {
      if (patients.length === 0) {
        setRecentScans([]);
        return;
      }

      const patientIds = patients.map((patient) => patient.id);
      const { data, error } = await supabase
        .from('medical_scans')
        .select('id, patient_id, created_at, scan_type, severity, impression')
        .in('patient_id', patientIds)
        .order('created_at', { ascending: false })
        .limit(8);

      if (!error && data) {
        setRecentScans(data as RecentScan[]);
      }
    };

    void loadRecentScans();
  }, [patients]);

  const doctorName = useMemo(() => {
    const fullName = doctorProfile?.full_name || user?.email?.split('@')[0] || 'Doctor';
    return fullName.replace(/^Dr\.?\s+/i, '');
  }, [doctorProfile, user]);

  const patientById = useMemo(() => {
    return patients.reduce<Record<string, Patient>>((acc, patient) => {
      acc[patient.id] = patient;
      return acc;
    }, {});
  }, [patients]);

  const lastPatient = useMemo(() => {
    if (selectedPatient) return selectedPatient;
    if (recentScans[0]) return patientById[recentScans[0].patient_id] || null;
    return [...patients].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())[0] || null;
  }, [patientById, patients, recentScans, selectedPatient]);

  const stats = useMemo(() => {
    const weekStart = startOfWeek();
    const studiesThisWeek = recentScans.filter((scan) => new Date(scan.created_at) >= weekStart).length;
    const criticalCases = patients.filter((patient) => patient.status === 'critical').length;
    const activeCases = patients.filter((patient) => patient.status === 'active').length;
    return {
      totalPatients: patients.length,
      activeCases,
      criticalCases,
      studiesThisWeek,
    };
  }, [patients, recentScans]);

  const openPatient = (patient: Patient | null, route = '/patients') => {
    if (patient) setSelectedPatient(patient);
    navigation.navigate(route);
  };

  const recentWorklist = recentScans.slice(0, 5);
  const isBusy = loading || patientsLoading;

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView showsVerticalScrollIndicator={false} contentInsetAdjustmentBehavior="automatic" contentContainerStyle={s.scroll}>
        <View style={s.heroCard}>
          <View style={s.badge}>
            <ShieldCheck color={theme.primary} size={14} />
            <Text style={s.badgeText}>Radiologist Workspace</Text>
          </View>
          <Text style={s.title}>
            {getGreeting()}, <Text style={s.titleAccent}>Dr. {doctorName}</Text>
          </Text>
          <Text style={s.copy}>Welcome back. Access your patient registry, start new diagnostic reports, and review active clinical evidence.</Text>
          <View style={s.actions}>
            <TouchableOpacity style={s.primaryButton} onPress={() => openPatient(lastPatient, lastPatient ? '/upload' : '/patients')}>
              <ImagePlus color={theme.primaryForeground} size={16} />
              <Text style={s.primaryButtonText}>{lastPatient ? `Continue ${lastPatient.name}` : 'Select patient'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={s.secondaryButton} onPress={() => navigation.navigate('/patients')}>
              <Search color={theme.foreground} size={16} />
              <Text style={s.secondaryButtonText}>Open patient registry</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={s.secondaryButton}
              onPress={async () => {
                await signOut();
                navigation.reset({ index: 0, routes: [{ name: '/login' }] });
              }}
            >
              <Text style={s.secondaryButtonText}>Log out</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={s.statsRow}>
          <MetricCard theme={theme} icon={Users} label="Patient Panel" value={stats.totalPatients} detail={`${stats.activeCases} active cases`} />
          <MetricCard theme={theme} icon={FileText} label="Studies This Week" value={stats.studiesThisWeek} detail="Recent generated reports" />
          <MetricCard theme={theme} icon={AlertTriangle} label="Critical Cases" value={stats.criticalCases} detail="Need close follow-up" tone="destructive" />
        </View>

        <View style={s.grid}>
          <View style={s.columnCard}>
            <View style={s.sectionHeader}>
              <UserRound color={theme.primary} size={18} />
              <Text style={s.sectionTitle}>Last Patient</Text>
            </View>
            {lastPatient ? (
              <>
                <View style={s.patientHighlight}>
                  <View style={s.patientHeader}>
                    <View>
                      <Text style={s.patientName}>{lastPatient.name}</Text>
                      <Text style={s.patientMeta}>{lastPatient.age} years / {lastPatient.gender}</Text>
                    </View>
                    <View style={s.patientStatusBadge}>
                      <Text style={s.patientStatusText}>{lastPatient.status}</Text>
                    </View>
                  </View>
                  <View style={s.conditionRow}>
                    {(lastPatient.conditions || []).slice(0, 3).map((condition) => (
                      <View key={condition} style={s.conditionChip}>
                        <Text style={s.conditionChipText}>{condition}</Text>
                      </View>
                    ))}
                  </View>
                </View>
                <View style={s.cardActions}>
                  <TouchableOpacity style={s.primaryButtonBlock} onPress={() => openPatient(lastPatient, '/upload')}>
                    <Text style={s.primaryButtonText}>Start report</Text>
                    <ArrowRight color={theme.primaryForeground} size={16} />
                  </TouchableOpacity>
                  <TouchableOpacity style={s.secondaryButtonBlock} onPress={() => openPatient(lastPatient, '/patients')}>
                    <Text style={s.secondaryButtonText}>Open history</Text>
                    <ArrowRight color={theme.foreground} size={16} />
                  </TouchableOpacity>
                </View>
              </>
            ) : (
              <EmptyState theme={theme} title="No patient selected" copy="Create or select a patient to start a radiology report workflow." action="Open registry" onAction={() => navigation.navigate('/patients')} />
            )}
          </View>

          <View style={s.columnCard}>
            <View style={s.sectionHeader}>
              <Activity color={theme.primary} size={18} />
              <Text style={s.sectionTitle}>Recent Worklist</Text>
            </View>
            {isBusy ? (
              <View style={s.loadingCard}>
                <ActivityIndicator color={theme.primary} />
              </View>
            ) : recentWorklist.length === 0 ? (
              <EmptyState theme={theme} title="No recent studies" copy="Upload a new scan to populate your report queue." action="Upload scan" onAction={() => navigation.navigate('/upload')} />
            ) : (
              <View style={s.worklist}>
                {recentWorklist.map((scan) => {
                  const patient = patientById[scan.patient_id];
                  return (
                    <TouchableOpacity
                      key={scan.id}
                      style={s.worklistItem}
                      onPress={() => {
                        if (patient) setSelectedPatient(patient);
                        navigation.navigate('/upload');
                      }}
                    >
                      <View style={{ flex: 1 }}>
                        <Text style={s.worklistPatient}>{patient?.name || scan.patient_id}</Text>
                        <Text style={s.worklistImpression}>{scan.impression || 'No impression yet'}</Text>
                      </View>
                      <Text style={s.worklistSeverity}>{scan.severity || 'moderate'}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            )}
          </View>

          <View style={s.columnCard}>
            <View style={s.sectionHeader}>
              <ShieldCheck color={theme.primary} size={18} />
              <Text style={s.sectionTitle}>HIL Tasks</Text>
            </View>
            {hilTasks.length === 0 ? (
              <EmptyState theme={theme} title="No active HIL tasks" copy="Assigned human-in-the-loop reviews will show up here." action="Open patients" onAction={() => navigation.navigate('/patients')} />
            ) : (
              <View style={s.worklist}>
                {hilTasks.map((task) => (
                  <TouchableOpacity key={task.id} style={s.worklistItem} onPress={() => navigation.navigate('/hil/task/:taskId', { taskId: task.id })}>
                    <View style={{ flex: 1 }}>
                      <Text style={s.worklistPatient}>{task.title}</Text>
                      <Text style={s.worklistImpression}>{task.completed_scans}/{task.total_scans} scans labeled</Text>
                    </View>
                    <ArrowRight color={theme.primary} size={16} />
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

function MetricCard({
  theme,
  icon: Icon,
  label,
  value,
  detail,
  tone = 'primary',
}: {
  theme: any;
  icon: any;
  label: string;
  value: number;
  detail: string;
  tone?: 'primary' | 'destructive';
}) {
  return (
    <View style={[styles(theme).metricCard, tone === 'destructive' && styles(theme).metricCardDanger]}>
      <Icon color={tone === 'destructive' ? theme.destructive : theme.primary} size={18} />
      <Text style={styles(theme).metricValue}>{value}</Text>
      <Text style={styles(theme).metricLabel}>{label}</Text>
      <Text style={styles(theme).metricDetail}>{detail}</Text>
    </View>
  );
}

function EmptyState({
  theme,
  title,
  copy,
  action,
  onAction,
}: {
  theme: any;
  title: string;
  copy: string;
  action: string;
  onAction: () => void;
}) {
  return (
    <View style={styles(theme).emptyState}>
      <Text style={styles(theme).emptyTitle}>{title}</Text>
      <Text style={styles(theme).emptyCopy}>{copy}</Text>
      <TouchableOpacity style={styles(theme).secondaryButtonBlock} onPress={onAction}>
        <Text style={styles(theme).secondaryButtonText}>{action}</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = (theme: any) => StyleSheet.create({
  container: { flex: 1, ...shellBackground(theme) },
  scroll: { paddingHorizontal: spacing.lg, paddingTop: spacing.xl, paddingBottom: spacing.xxl + 84, gap: spacing.lg },
  heroCard: { ...glassPanel(theme), padding: spacing.lg, gap: spacing.md },
  badge: { ...mutedCard(theme), alignSelf: 'flex-start', flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: spacing.sm, paddingVertical: 6 },
  badgeText: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
  title: { color: theme.foreground, fontSize: typography['2xl'], fontFamily: fontFamily.extraBold, lineHeight: 34 },
  titleAccent: { color: theme.primary },
  copy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
  actions: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap' },
  primaryButton: { backgroundColor: theme.primary, borderRadius: radius.lg, height: 48, paddingHorizontal: spacing.md, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, ...createShadow(theme, 'md') },
  primaryButtonText: { color: theme.primaryForeground, fontSize: typography.sm, fontFamily: fontFamily.bold },
  secondaryButton: { ...surfaceCard(theme), height: 48, paddingHorizontal: spacing.md, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  secondaryButtonText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
  statsRow: { gap: spacing.sm },
  metricCard: { ...surfaceCard(theme), padding: spacing.md, gap: 4 },
  metricCardDanger: { borderColor: theme.destructive },
  metricValue: { color: theme.foreground, fontSize: typography.xl, fontFamily: fontFamily.extraBold },
  metricLabel: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.bold },
  metricDetail: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  grid: { gap: spacing.md },
  columnCard: { ...surfaceCard(theme), padding: spacing.md, gap: spacing.md },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  sectionTitle: { color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.bold },
  patientHighlight: { borderWidth: 1, borderColor: theme.primaryGlow, borderRadius: radius.xl, backgroundColor: theme.primaryGlow, padding: spacing.md, gap: spacing.md },
  patientHeader: { flexDirection: 'row', justifyContent: 'space-between', gap: spacing.md },
  patientName: { color: theme.foreground, fontSize: typography.lg, fontFamily: fontFamily.extraBold },
  patientMeta: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.regular, marginTop: 2 },
  patientStatusBadge: { borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.full, paddingHorizontal: spacing.sm, paddingVertical: 4, backgroundColor: theme.card },
  patientStatusText: { color: theme.foreground, fontSize: typography.xs, fontFamily: fontFamily.semiBold, textTransform: 'capitalize' },
  conditionRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  conditionChip: { borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.full, backgroundColor: theme.card, paddingHorizontal: spacing.sm, paddingVertical: 4 },
  conditionChipText: { color: theme.foreground, fontSize: typography.xs, fontFamily: fontFamily.semiBold },
  cardActions: { gap: spacing.sm },
  primaryButtonBlock: { backgroundColor: theme.primary, borderRadius: radius.lg, height: 48, paddingHorizontal: spacing.md, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', ...createShadow(theme, 'sm') },
  secondaryButtonBlock: { ...surfaceCard(theme), height: 48, paddingHorizontal: spacing.md, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  loadingCard: { paddingVertical: spacing.lg, alignItems: 'center', justifyContent: 'center' },
  worklist: { gap: spacing.sm },
  worklistItem: { borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.lg, backgroundColor: theme.card, padding: spacing.md, flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  worklistPatient: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.bold },
  worklistImpression: { color: theme.mutedForeground, fontSize: typography.xs, lineHeight: 18, fontFamily: fontFamily.regular, marginTop: 2 },
  worklistSeverity: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.semiBold, textTransform: 'capitalize' },
  emptyState: { gap: spacing.sm },
  emptyTitle: { color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.bold },
  emptyCopy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 20, fontFamily: fontFamily.regular },
});
