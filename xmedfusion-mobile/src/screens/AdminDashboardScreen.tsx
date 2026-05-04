import React, { useEffect, useMemo, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, StatusBar, ActivityIndicator } from 'react-native';
import { Activity, Brain, Clock, Cpu, Database, FileText, Plus, ShieldCheck, Users } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { glassPanel, shellBackground, surfaceCard } from '../theme/ui';
import { useAuth } from '../theme/AuthContext';
import { supabase } from '../lib/supabase';
import { getApiBase } from '../lib/apiConfig';

interface PendingRequest {
  user_id: string;
  created_at: string;
  doctor?: {
    full_name: string;
    email: string;
    specialization: string;
  } | null;
}

interface HilTaskSummary {
  id: string;
  title: string;
  total_scans: number;
  completed_scans: number;
  status: string;
}

interface HealthSummary {
  status: string;
  uptime_seconds?: number;
  cpu_percent?: number;
  memory_used_mb?: number;
  memory_total_mb?: number;
  gpu_available?: boolean;
  ai_warmup_complete?: boolean;
}

const formatUptime = (seconds?: number) => {
  if (!seconds) return 'Unavailable';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
};

export default function AdminDashboardScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const { user, isAdmin, signOut } = useAuth();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState([
    { label: 'Doctors', value: 0, icon: Users },
    { label: 'Patients', value: 0, icon: Activity },
    { label: 'Reports', value: 0, icon: FileText },
    { label: 'HIL Tasks', value: 0, icon: Brain },
  ]);
  const [pendingRequests, setPendingRequests] = useState<PendingRequest[]>([]);
  const [hilTasks, setHilTasks] = useState<HilTaskSummary[]>([]);
  const [health, setHealth] = useState<HealthSummary | null>(null);
  const s = styles(theme);

  useEffect(() => {
    if (!user) {
      navigation.reset({ index: 0, routes: [{ name: '/login' }] });
      return;
    }

    if (!isAdmin) {
      navigation.reset({ index: 0, routes: [{ name: '/dashboard' }] });
      return;
    }

    const loadAdminData = async () => {
      setLoading(true);
      try {
        const [
          doctorsCount,
          patientsCount,
          reportsCount,
          tasksResponse,
          pendingRoles,
          doctorsDirectory,
        ] = await Promise.all([
          supabase.from('doctors').select('id', { count: 'exact', head: true }),
          supabase.from('patients').select('id', { count: 'exact', head: true }),
          supabase.from('medical_scans').select('id', { count: 'exact', head: true }),
          supabase
            .from('hil_tasks')
            .select('id, title, total_scans, completed_scans, status')
            .order('created_at', { ascending: false })
            .limit(4),
          supabase
            .from('user_roles')
            .select('user_id, created_at')
            .eq('approval_status', 'pending')
            .order('created_at', { ascending: false })
            .limit(5),
          supabase
            .from('doctors')
            .select('user_id, full_name, email, specialization'),
        ]);

        setStats([
          { label: 'Doctors', value: doctorsCount.count ?? 0, icon: Users },
          { label: 'Patients', value: patientsCount.count ?? 0, icon: Activity },
          { label: 'Reports', value: reportsCount.count ?? 0, icon: FileText },
          { label: 'HIL Tasks', value: (tasksResponse.data || []).length, icon: Brain },
        ]);
        setHilTasks((tasksResponse.data || []) as HilTaskSummary[]);

        const doctorMap = new Map(
          (doctorsDirectory.data || []).map((doctor) => [
            doctor.user_id,
            {
              full_name: doctor.full_name,
              email: doctor.email,
              specialization: doctor.specialization,
            },
          ])
        );

        const mergedPending = (pendingRoles.data || []).map((role) => ({
          ...role,
          doctor: doctorMap.get(role.user_id) || null,
        }));
        setPendingRequests(mergedPending as PendingRequest[]);

        try {
          const apiBase = await getApiBase(true);
          const response = await fetch(`${apiBase}/api/health`, {
            headers: { 'ngrok-skip-browser-warning': 'true' },
          });
          if (response.ok) {
            const payload = await response.json();
            setHealth(payload as HealthSummary);
          } else {
            setHealth(null);
          }
        } catch {
          setHealth(null);
        }
      } finally {
        setLoading(false);
      }
    };

    void loadAdminData();
  }, [isAdmin, navigation, user]);

  const healthStatusText = useMemo(() => {
    if (!health) return 'Backend offline or unreachable';
    if (health.status === 'healthy') return 'Backend healthy';
    return `Backend ${health.status}`;
  }, [health]);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView contentInsetAdjustmentBehavior="automatic" contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.heroCard}>
          <View style={s.badge}>
            <ShieldCheck color={theme.primary} size={14} />
            <Text style={s.badgeText}>Admin Dashboard</Text>
          </View>
          <Text style={s.title}>Platform management and HIL oversight</Text>
          <Text style={s.copy}>Doctor registry, pending approvals, system health, and human-in-the-loop tasks are surfaced here just like the web app.</Text>
          <View style={s.actions}>
            <TouchableOpacity style={s.primaryButton} onPress={() => navigation.navigate('/pending')}>
              <Plus color={theme.primaryForeground} size={16} />
              <Text style={s.primaryButtonText}>Review Pending</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={s.secondaryButton}
              onPress={() => navigation.navigate('/hil/task/:taskId', { taskId: hilTasks[0]?.id || 'demo' })}
            >
              <FileText color={theme.foreground} size={16} />
              <Text style={s.secondaryButtonText}>Open HIL Task</Text>
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
          {stats.map((stat) => (
            <View key={stat.label} style={s.statCard}>
              <stat.icon color={theme.primary} size={20} />
              <Text style={s.statValue}>{String(stat.value)}</Text>
              <Text style={s.statLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>

        <View style={s.sectionCard}>
          <Text style={s.sectionTitle}>Server Health</Text>
          <View style={s.row}>
            <Cpu color={theme.primary} size={18} />
            <Text style={s.rowText}>
              {health ? `CPU ${Math.round(health.cpu_percent || 0)}% - RAM ${health.memory_used_mb || 0} MB / ${health.memory_total_mb || 0} MB` : 'Unable to read live server metrics'}
            </Text>
          </View>
          <View style={s.row}>
            <Database color={theme.primary} size={18} />
            <Text style={s.rowText}>{healthStatusText}</Text>
          </View>
          <View style={s.row}>
            <Clock color={theme.primary} size={18} />
            <Text style={s.rowText}>
              Uptime {formatUptime(health?.uptime_seconds)} - AI warmup {health?.ai_warmup_complete ? 'ready' : 'pending'}
            </Text>
          </View>
        </View>

        <View style={s.sectionCard}>
          <Text style={s.sectionTitle}>Pending Approvals</Text>
          {loading ? (
            <View style={s.loadingState}>
              <ActivityIndicator color={theme.primary} />
            </View>
          ) : pendingRequests.length === 0 ? (
            <Text style={s.rowText}>No pending doctor approvals right now.</Text>
          ) : (
            pendingRequests.map((request) => (
              <View key={request.user_id} style={s.listItem}>
                <View style={{ flex: 1 }}>
                  <Text style={s.listTitle}>{request.doctor?.full_name || request.user_id}</Text>
                  <Text style={s.rowText}>{request.doctor?.email || 'Awaiting doctor profile completion'}</Text>
                </View>
                <Text style={s.listMeta}>{request.doctor?.specialization || 'Pending'}</Text>
              </View>
            ))
          )}
        </View>

        <View style={s.sectionCard}>
          <Text style={s.sectionTitle}>Recent HIL Tasks</Text>
          {loading ? (
            <View style={s.loadingState}>
              <ActivityIndicator color={theme.primary} />
            </View>
          ) : hilTasks.length === 0 ? (
            <Text style={s.rowText}>No HIL tasks have been created yet.</Text>
          ) : (
            hilTasks.map((task) => (
              <TouchableOpacity
                key={task.id}
                style={s.listItem}
                onPress={() => navigation.navigate('/hil/task/:taskId', { taskId: task.id })}
              >
                <View style={{ flex: 1 }}>
                  <Text style={s.listTitle}>{task.title}</Text>
                  <Text style={s.rowText}>{task.completed_scans}/{task.total_scans} scans reviewed</Text>
                </View>
                <Text style={s.listMeta}>{task.status.replace('_', ' ')}</Text>
              </TouchableOpacity>
            ))
          )}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) =>
  StyleSheet.create({
    container: { flex: 1, ...shellBackground(theme) },
    scroll: { paddingHorizontal: spacing.lg, paddingTop: spacing.xl, paddingBottom: spacing.xxl + 84, gap: spacing.lg },
    heroCard: { ...glassPanel(theme), padding: spacing.lg, gap: spacing.md },
    badge: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 6, borderRadius: radius.full, backgroundColor: theme.primaryGlow },
    badgeText: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
    title: { color: theme.foreground, fontSize: typography['2xl'], fontFamily: fontFamily.extraBold },
    copy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
    actions: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap' },
    primaryButton: { backgroundColor: theme.primary, borderRadius: radius.lg, height: 50, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingHorizontal: spacing.md },
    primaryButtonText: { color: theme.primaryForeground, fontSize: typography.sm, fontFamily: fontFamily.bold },
    secondaryButton: { ...surfaceCard(theme), height: 50, paddingHorizontal: spacing.md, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
    secondaryButtonText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
    statsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
    statCard: { ...surfaceCard(theme), width: '48%', padding: spacing.md, gap: 4 },
    statValue: { color: theme.foreground, fontSize: typography.lg, fontFamily: fontFamily.extraBold },
    statLabel: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
    sectionCard: { ...surfaceCard(theme), padding: spacing.md, gap: spacing.md },
    sectionTitle: { color: theme.foreground, fontSize: typography.lg, fontFamily: fontFamily.bold },
    row: { flexDirection: 'row', alignItems: 'center', gap: 10 },
    rowText: { color: theme.mutedForeground, flex: 1, fontSize: typography.sm, fontFamily: fontFamily.regular },
    loadingState: { paddingVertical: spacing.md, alignItems: 'center' },
    listItem: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, paddingVertical: spacing.xs },
    listTitle: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.bold },
    listMeta: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.semiBold, textTransform: 'capitalize' },
  });
