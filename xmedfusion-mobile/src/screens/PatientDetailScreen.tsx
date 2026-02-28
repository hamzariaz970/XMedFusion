import React from 'react';
import { View, Text, StyleSheet, ScrollView, StatusBar, TouchableOpacity, FlatList } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import { 
  ChevronLeft, 
  User, 
  Phone, 
  Mail, 
  Calendar, 
  FileText, 
  ShieldAlert, 
  ShieldCheck, 
  ChevronRight,
  Activity,
  UserCheck
} from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

const clinicalHistory = [
  { id: 'H001', date: 'Feb 28, 2026', title: 'Chest X-Ray (PA)', status: 'critical', finding: 'Bilateral Consolidation' },
  { id: 'H002', date: 'Jan 15, 2026', title: 'ECG Analysis', status: 'normal', finding: 'Normal Sinus Rhythm' },
  { id: 'H003', date: 'Nov 10, 2025', title: 'Chest X-Ray (Lat)', status: 'warning', finding: 'Mild Cardiomegaly' },
];

export default function PatientDetailScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const { theme, isDark } = useTheme();
  const s = styles(theme);

  const patient = route.params?.patient || {
    id: 'PT-2847',
    name: 'Ali Hassan',
    age: 52,
    gender: 'Male',
    contact: '+92 321 4455663',
    email: 'ali.hassan@email.com',
    bloodType: 'O+'
  };

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      
      {/* Header */}
      <View style={s.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <ChevronLeft color={theme.foreground} size={24} />
        </TouchableOpacity>
        <Text style={s.headerTitle}>Patient Profile</Text>
        <TouchableOpacity onPress={() => alert('Patient marked as Verified.')}>
           <UserCheck color={theme.primary} size={24} />
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        {/* Profile Card */}
        <View style={s.profileCard}>
           <View style={s.avatarLarge}>
              <User color={theme.primary} size={48} strokeWidth={1.5} />
           </View>
           <Text style={s.patientName}>{patient.name}</Text>
           <Text style={s.patientId}>{patient.id} · {patient.age}y · {patient.gender}</Text>
           
           <View style={s.statsRow}>
              <View style={s.statBox}>
                 <Text style={s.statValue}>{patient.bloodType}</Text>
                 <Text style={s.statLabel}>Blood</Text>
              </View>
              <View style={s.statDivider} />
              <View style={s.statBox}>
                 <Text style={s.statValue}>82</Text>
                 <Text style={s.statLabel}>Heart Rate</Text>
              </View>
              <View style={s.statDivider} />
              <View style={s.statBox}>
                 <Text style={s.statValue}>128/85</Text>
                 <Text style={s.statLabel}>BP</Text>
              </View>
           </View>
        </View>

        {/* Contact Info */}
        <View style={s.infoCard}>
           <View style={s.infoRow}>
              <Phone color={theme.mutedForeground} size={18} />
              <Text style={s.infoText}>{patient.contact}</Text>
           </View>
           <View style={s.infoRow}>
              <Mail color={theme.mutedForeground} size={18} />
              <Text style={s.infoText}>{patient.email}</Text>
           </View>
        </View>

        {/* Clinical Tabs */}
        <View style={s.tabsContainer}>
           <Text style={s.sectionTitle}>Case History</Text>
           <TouchableOpacity><Text style={s.seeAll}>Timeline</Text></TouchableOpacity>
        </View>

        {clinicalHistory.map((item) => (
          <TouchableOpacity 
            key={item.id} 
            style={s.historyCard} 
            activeOpacity={0.8}
            onPress={() => navigation.navigate('ReportDetail', { report: { ...item, patient: patient.name } })}
          >
            <View style={[s.historyIcon, { backgroundColor: item.status === 'critical' ? theme.destructiveBg : item.status === 'warning' ? theme.warningBg : theme.successBg }]}>
               {item.status === 'critical' ? <ShieldAlert color={theme.destructive} size={20} /> : item.status === 'warning' ? <Activity color={theme.warning} size={20} /> : <ShieldCheck color={theme.success} size={20} />}
            </View>
            <View style={s.historyMain}>
               <Text style={s.historyTitle}>{item.title}</Text>
               <Text style={s.historyDate}>{item.date}</Text>
            </View>
            <View style={s.historyRight}>
               <Text style={[s.historyFinding, { color: item.status === 'critical' ? theme.destructive : theme.mutedForeground }]}>{item.finding}</Text>
               <ChevronRight color={theme.mutedForeground} size={16} />
            </View>
          </TouchableOpacity>
        ))}

        <TouchableOpacity style={s.newReportBtn} onPress={() => navigation.navigate('Upload')}>
           <FileText color={theme.primaryForeground} size={18} />
           <Text style={s.newReportText}>Start New Analysis</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.background },
  header: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    justifyContent: 'space-between', 
    paddingHorizontal: spacing.lg, 
    paddingTop: spacing.xxl, 
    paddingBottom: spacing.md,
    backgroundColor: theme.card,
    borderBottomWidth: 1,
    borderBottomColor: theme.cardBorder
  },
  headerTitle: { color: theme.foreground, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold },
  scroll: { padding: spacing.lg, gap: spacing.lg, paddingBottom: spacing.xxl },
  
  profileCard: { backgroundColor: theme.card, borderRadius: radius.lg, padding: spacing.xl, alignItems: 'center', gap: 6, borderWidth: 1, borderColor: theme.cardBorder },
  avatarLarge: { width: 80, height: 80, borderRadius: 40, backgroundColor: theme.primaryGlow, alignItems: 'center', justifyContent: 'center', marginBottom: 8 },
  patientName: { color: theme.foreground, fontSize: typography['2xl'], fontWeight: '700', fontFamily: fontFamily.bold },
  patientId: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.medium },
  
  statsRow: { flexDirection: 'row', alignItems: 'center', marginTop: spacing.lg, width: '100%' },
  statBox: { flex: 1, alignItems: 'center' },
  statValue: { color: theme.foreground, fontSize: typography.base, fontWeight: '700', fontFamily: fontFamily.bold },
  statLabel: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  statDivider: { width: 1, height: 30, backgroundColor: theme.cardBorder },

  infoCard: { backgroundColor: theme.card, borderRadius: radius.md, padding: spacing.md, gap: spacing.md, borderWidth: 1, borderColor: theme.cardBorder },
  infoRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  infoText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.medium },

  tabsContainer: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: spacing.sm },
  sectionTitle: { color: theme.foreground, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold },
  seeAll: { color: theme.primary, fontSize: typography.sm, fontWeight: '600', fontFamily: fontFamily.semiBold },

  historyCard: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.card, borderRadius: radius.md, padding: spacing.md, gap: spacing.md, borderWidth: 1, borderColor: theme.cardBorder },
  historyIcon: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
  historyMain: { flex: 1 },
  historyTitle: { color: theme.foreground, fontWeight: '600', fontSize: typography.sm, fontFamily: fontFamily.semiBold },
  historyDate: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  historyRight: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  historyFinding: { fontSize: typography.xs, fontFamily: fontFamily.medium, maxWidth: 100, textAlign: 'right' },

  newReportBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: theme.primary, height: 52, borderRadius: radius.lg, marginTop: spacing.sm },
  newReportText: { color: theme.primaryForeground, fontWeight: '700', fontSize: typography.sm, fontFamily: fontFamily.bold },
});
