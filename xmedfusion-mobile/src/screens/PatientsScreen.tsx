import React, { useEffect, useMemo, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, ScrollView, TextInput, ActivityIndicator } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Activity, AlertTriangle, CheckCircle, Clock, FileText, Search, ShieldCheck, Sparkles, Stethoscope, User } from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { glassPanel, mutedCard, shellBackground, surfaceCard } from '../theme/ui';
import { Patient, usePatientContext } from '../context/PatientContext';
import { useAuth } from '../theme/AuthContext';
import { supabase } from '../lib/supabase';

interface HilTaskSummary {
  id: string;
  title: string;
  total_scans: number;
  completed_scans: number;
}

export default function PatientsScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const { patients, selectedPatient, setSelectedPatient, loading } = usePatientContext();
  const { user } = useAuth();
  const [search, setSearch] = useState('');
  const [hilTasks, setHilTasks] = useState<HilTaskSummary[]>([]);
  const s = styles(theme);

  useEffect(() => {
    if (!user) return;

    const fetchHil = async () => {
      const { data } = await supabase
        .from('hil_tasks')
        .select('id, title, total_scans, completed_scans')
        .eq('doctor_id', user.id)
        .in('status', ['assigned', 'in_progress'])
        .order('created_at', { ascending: false });

      setHilTasks((data || []) as HilTaskSummary[]);
    };

    void fetchHil();
  }, [user]);

  const filtered = useMemo(
    () =>
      patients.filter(
        (patient) => patient.name.toLowerCase().includes(search.toLowerCase()) || patient.id.toLowerCase().includes(search.toLowerCase())
      ),
    [patients, search]
  );

  const openPatient = (patient: Patient) => {
    setSelectedPatient(patient);
    navigation.navigate('/upload');
  };

  const statusIcon = (status: string) => {
    if (status === 'critical') return <AlertTriangle color={theme.destructive} size={12} />;
    if (status === 'resolved') return <CheckCircle color={theme.success} size={12} />;
    return <Activity color={theme.primary} size={12} />;
  };

  const statusColor = (status: string) =>
    status === 'critical' ? theme.destructive : status === 'resolved' ? theme.success : theme.primary;

  const statusBg = (status: string) =>
    status === 'critical' ? theme.destructiveBg : status === 'resolved' ? theme.successBg : theme.primaryGlow;

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView contentInsetAdjustmentBehavior="automatic" contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.heroCard}>
          <View style={s.badge}>
            <ShieldCheck color={theme.primary} size={14} />
            <Text style={s.badgeText}>Clinical case registry</Text>
          </View>
          <Text style={s.title}>Patient Dashboard</Text>
          <Text style={s.copy}>Track and manage patient cases and diagnostic history.</Text>
          <View style={s.heroPill}>
            <Sparkles color={theme.primary} size={14} />
            <Text style={s.heroPillText}>{patients.length} patients in registry</Text>
          </View>
        </View>

        {hilTasks.length > 0 && (
          <View style={s.hilCard}>
            <View style={s.sectionHeader}>
              <Stethoscope color={theme.primary} size={18} />
              <Text style={s.sectionTitle}>Pending Labeling Tasks</Text>
            </View>
            <Text style={s.sectionCopy}>You have {hilTasks.length} task(s) awaiting your expert review.</Text>
            <View style={s.hilList}>
              {hilTasks.map((task) => (
                <TouchableOpacity key={task.id} style={s.hilItem} onPress={() => navigation.navigate('/hil/task/:taskId', { taskId: task.id })}>
                  <View style={{ flex: 1 }}>
                    <Text style={s.hilItemTitle}>{task.title}</Text>
                    <Text style={s.hilItemMeta}>{task.completed_scans}/{task.total_scans} scans labeled</Text>
                  </View>
                  <FileText color={theme.primary} size={16} />
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        <View style={s.searchCard}>
          <Search color={theme.mutedForeground} size={18} />
          <TextInput
            style={s.searchInput}
            placeholder="Search patients..."
            placeholderTextColor={theme.mutedForeground}
            value={search}
            onChangeText={setSearch}
          />
        </View>

        {selectedPatient ? (
          <View style={s.selectedBanner}>
            <Text style={s.selectedBannerText}>Selected patient: {selectedPatient.name}</Text>
          </View>
        ) : null}

        {loading ? (
          <View style={s.loadingCard}>
            <ActivityIndicator color={theme.primary} />
          </View>
        ) : (
          <View style={s.list}>
            {filtered.map((patient) => (
              <TouchableOpacity key={patient.id} style={s.patientCard} activeOpacity={0.85} onPress={() => openPatient(patient)}>
                <View style={[s.avatar, { backgroundColor: statusBg(patient.status) }]}>
                  <User color={statusColor(patient.status)} size={20} />
                </View>
                <View style={s.info}>
                  <Text style={s.patientName}>{patient.name}</Text>
                  <Text style={s.patientMeta}>{patient.age} / {patient.gender}</Text>
                  <View style={s.conditionRow}>
                    {(patient.conditions || []).slice(0, 2).map((condition) => (
                      <View key={condition} style={s.conditionChip}>
                        <Text style={s.conditionChipText}>{condition}</Text>
                      </View>
                    ))}
                  </View>
                </View>
                <View style={s.right}>
                  <View style={[s.statusBadge, { backgroundColor: statusBg(patient.status) }]}>
                    {statusIcon(patient.status)}
                    <Text style={[s.statusText, { color: statusColor(patient.status) }]}>{patient.status}</Text>
                  </View>
                  <View style={s.timeRow}>
                    <Clock color={theme.mutedForeground} size={12} />
                    <Text style={s.lastReport}>{new Date(patient.updated_at).toLocaleDateString()}</Text>
                  </View>
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) =>
  StyleSheet.create({
    container: { flex: 1, ...shellBackground(theme) },
    scroll: { paddingHorizontal: spacing.lg, paddingTop: spacing.xl, paddingBottom: spacing.xxl + 84, gap: spacing.lg },
    heroCard: { ...glassPanel(theme), padding: spacing.lg, gap: spacing.sm },
    badge: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 6, borderRadius: radius.full, backgroundColor: theme.primaryGlow },
    badgeText: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
    title: { color: theme.foreground, fontSize: typography['2xl'], fontFamily: fontFamily.extraBold },
    copy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
    heroPill: { ...mutedCard(theme), alignSelf: 'flex-start', flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: spacing.sm, paddingVertical: 8 },
    heroPillText: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.bold },
    hilCard: { ...surfaceCard(theme), padding: spacing.md, gap: spacing.sm },
    sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    sectionTitle: { color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.bold },
    sectionCopy: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.regular },
    hilList: { gap: spacing.sm },
    hilItem: { borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.lg, backgroundColor: theme.card, padding: spacing.md, flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
    hilItemTitle: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.bold },
    hilItemMeta: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular, marginTop: 2 },
    searchCard: { ...surfaceCard(theme), flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.md, gap: 10 },
    searchInput: { flex: 1, height: 50, color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.regular },
    selectedBanner: { ...mutedCard(theme), paddingHorizontal: spacing.md, paddingVertical: 10 },
    selectedBannerText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
    loadingCard: { paddingVertical: spacing.xl, alignItems: 'center' },
    list: { gap: spacing.sm },
    patientCard: { flexDirection: 'row', alignItems: 'center', ...surfaceCard(theme), padding: spacing.md, gap: spacing.md },
    avatar: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
    info: { flex: 1, gap: 3 },
    patientName: { color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.bold },
    patientMeta: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
    conditionRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 2 },
    conditionChip: { borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.full, backgroundColor: theme.background, paddingHorizontal: spacing.sm, paddingVertical: 4 },
    conditionChipText: { color: theme.foreground, fontSize: typography.xs, fontFamily: fontFamily.semiBold },
    right: { alignItems: 'flex-end', gap: 4 },
    statusBadge: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: spacing.sm, paddingVertical: 4, borderRadius: radius.full },
    statusText: { fontSize: typography.xs, textTransform: 'capitalize', fontFamily: fontFamily.semiBold },
    timeRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
    lastReport: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  });
