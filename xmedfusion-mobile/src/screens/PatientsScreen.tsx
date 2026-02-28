import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, ScrollView, TextInput } from 'react-native';
import { 
  Search, 
  User, 
  ShieldAlert, 
  ShieldCheck, 
  Activity,
  History
} from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

const patients = [
  { id: 'PT-2847', name: 'Ali Hassan', age: 52, lastReport: '2h ago', status: 'critical', reports: 3 },
  { id: 'PT-2846', name: 'Ayesha Khan', age: 38, lastReport: '4h ago', status: 'warning', reports: 1 },
  { id: 'PT-2845', name: 'Omar Siddiqui', age: 67, lastReport: '6h ago', status: 'normal', reports: 5 },
  { id: 'PT-2844', name: 'Sara Malik', age: 44, lastReport: '1d ago', status: 'normal', reports: 2 },
  { id: 'PT-2843', name: 'Bilal Ahmed', age: 29, lastReport: '2d ago', status: 'warning', reports: 4 },
];

export default function PatientsScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const [search, setSearch] = useState('');
  const s = styles(theme);

  const severityColor = (sev: string) =>
    sev === 'critical' ? theme.destructive : sev === 'warning' ? theme.warning : theme.success;
  const severityBg = (sev: string) =>
    sev === 'critical' ? theme.destructiveBg : sev === 'warning' ? theme.warningBg : theme.successBg;

  const filtered = patients.filter(
    (p) => p.name.toLowerCase().includes(search.toLowerCase()) || p.id.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <View style={s.header}>
        <Text style={s.title}>Case Library</Text>
        <Text style={s.subtitle}>{patients.length} active patients</Text>
      </View>
      <View style={s.searchContainer}>
        <View style={s.searchWrapper}>
          <Search color={theme.mutedForeground} size={18} style={s.searchIcon} />
          <TextInput 
            style={s.searchInput} 
            placeholder="Search by ID or name..." 
            placeholderTextColor={theme.mutedForeground} 
            value={search} 
            onChangeText={setSearch} 
          />
        </View>
      </View>
      <ScrollView contentContainerStyle={s.list} showsVerticalScrollIndicator={false}>
        {filtered.map((patient) => (
          <TouchableOpacity 
            key={patient.id} 
            style={s.patientCard} 
            activeOpacity={0.8}
            onPress={() => navigation.navigate('PatientDetail', { patient })}
          >
            <View style={[s.avatar, { backgroundColor: severityBg(patient.status) }]}>
              <User color={severityColor(patient.status)} size={24} strokeWidth={2.5} />
            </View>
            <View style={s.info}>
              <Text style={s.patientName}>{patient.name}</Text>
              <Text style={s.patientMeta}>{patient.id} · Age {patient.age}</Text>
            </View>
            <View style={s.right}>
              <View style={[s.statusBadge, { backgroundColor: severityBg(patient.status) }]}>
                {patient.status === 'critical' ? (
                  <ShieldAlert color={theme.destructive} size={12} />
                ) : patient.status === 'warning' ? (
                  <Activity color={theme.warning} size={12} />
                ) : (
                  <ShieldCheck color={theme.success} size={12} />
                )}
                <Text style={[s.statusText, { color: severityColor(patient.status) }]}>{patient.status}</Text>
              </View>
              <View style={s.timeRow}>
                <History color={theme.mutedForeground} size={12} />
                <Text style={s.lastReport}>{patient.lastReport}</Text>
              </View>
            </View>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = (theme: ReturnType<typeof useTheme>['theme']) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.background },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.xxl, paddingBottom: spacing.md },
  title: { color: theme.foreground, fontSize: typography['2xl'], fontWeight: '700', fontFamily: fontFamily.bold },
  subtitle: { color: theme.mutedForeground, fontSize: typography.sm, marginTop: 2, fontFamily: fontFamily.regular },
  searchContainer: { paddingHorizontal: spacing.lg, marginBottom: spacing.md },
  searchWrapper: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.card, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.lg, paddingHorizontal: spacing.md },
  searchIcon: { marginRight: spacing.xs },
  searchInput: { flex: 1, height: 48, color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.regular },
  list: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl, gap: spacing.sm },
  patientCard: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.card, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.md, padding: spacing.md, gap: spacing.md },
  avatar: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
  info: { flex: 1 },
  patientName: { color: theme.foreground, fontWeight: '600', fontSize: typography.base, fontFamily: fontFamily.semiBold },
  patientMeta: { color: theme.mutedForeground, fontSize: typography.xs, marginTop: 2, fontFamily: fontFamily.regular },
  right: { alignItems: 'flex-end', gap: 4 },
  statusBadge: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: radius.full },
  statusText: { fontSize: typography.xs, fontWeight: '600', textTransform: 'capitalize', fontFamily: fontFamily.semiBold },
  timeRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  lastReport: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
});
