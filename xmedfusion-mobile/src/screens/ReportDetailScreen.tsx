import React from 'react';
import { View, Text, StyleSheet, ScrollView, StatusBar, TouchableOpacity } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import { 
  ChevronLeft, 
  Share, 
  ShieldCheck, 
  Activity, 
  Search, 
  AlertCircle,
  Clock,
  User,
  ExternalLink,
  ChevronDown,
  ChevronUp
} from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

export default function ReportDetailScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const { theme, isDark } = useTheme();
  const s = styles(theme);

  // Fallback data if route params are empty
  const report = route.params?.report || {
    id: 'RPT-8821',
    patient: 'PT-2847 (Ali Hassan)',
    status: 'Critical',
    time: 'Feb 28, 2026 · 14:30',
    finding: 'Bilateral Consolidation',
    confidence: 98.4
  };

  const sections = [
    {
      title: 'Automated Findings',
      icon: Search,
      content: 'Bilateral perihilar infiltrates are noted with increased opacity in the right lower lobe consistent with consolidation. The cardiomediastinal silhouette is mildly enlarged.'
    },
    {
      title: 'Clinical Impression',
      icon: Activity,
      content: 'Findings are consistent with bilateral pneumonia with early consolidative changes. Mild cardiomegaly noted. Clinical correlation with bedside ultrasound recommended.'
    },
    {
      title: 'AI Verification',
      icon: ShieldCheck,
      content: 'Verified by XMed-CXR Pipeline (v2.1). Confidence intervals: Pneumonia (98.4%), Atelectasis (12.2%), Effusion (3.5%). High visual saliency detected in Right Lower Lobe.'
    }
  ];

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      
      {/* Header */}
      <View style={s.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={s.backBtn}>
          <ChevronLeft color={theme.foreground} size={24} />
        </TouchableOpacity>
        <Text style={s.headerTitle}>Analysis Report</Text>
        <TouchableOpacity style={s.shareBtn} onPress={() => alert('Sharing options coming soon in this demo.')}>
          <Share color={theme.foreground} size={20} />
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        {/* Case Badge */}
        <View style={s.metaCard}>
          <View style={s.metaHeader}>
             <View style={s.idBadge}>
                <Text style={s.idText}>{report.id}</Text>
             </View>
             <View style={[s.statusBadge, { backgroundColor: theme.destructiveBg }]}>
                <AlertCircle color={theme.destructive} size={12} />
                <Text style={[s.statusText, { color: theme.destructive }]}>{report.status}</Text>
             </View>
          </View>
          <Text style={s.findingTitle}>{report.finding}</Text>
          <View style={s.metaRow}>
             <User color={theme.mutedForeground} size={14} />
             <Text style={s.metaText}>{report.patient}</Text>
             <View style={s.dot} />
             <Clock color={theme.mutedForeground} size={14} />
             <Text style={s.metaText}>{report.time}</Text>
          </View>
        </View>

        {/* X-Ray Visualization Placeholder */}
        <View style={s.vizPlaceholder}>
           <Search color={theme.primary} size={40} strokeWidth={1} style={{ marginBottom: 12 }} />
           <Text style={s.vizText}>Full-Resolution Radiograph Display</Text>
           <Text style={s.vizSubtext}>Interactive heatmap & saliency mapping available on desktop</Text>
           <TouchableOpacity style={s.expandBtn} onPress={() => alert('Full Medical Image Viewer is available on the Desktop Web interface.')}>
              <ExternalLink color={theme.primary} size={16} />
              <Text style={s.expandText}>Open Full Viewer</Text>
           </TouchableOpacity>
        </View>

        {/* Confidence Score */}
        <View style={s.confidenceCard}>
           <View style={s.confidenceHeader}>
              <Text style={s.confidenceLabel}>AI Confidence Score</Text>
              <Text style={s.confidenceValue}>{report.confidence}%</Text>
           </View>
           <View style={s.barBg}>
              <View style={[s.barFill, { width: `${report.confidence}%` }]} />
           </View>
        </View>

        {/* Detail Sections */}
        {sections.map((section, ix) => (
          <View key={ix} style={s.reportSection}>
            <View style={s.sectionHeader}>
               <section.icon color={theme.primary} size={20} />
               <Text style={s.sectionTitle}>{section.title}</Text>
            </View>
            <Text style={s.sectionContent}>{section.content}</Text>
          </View>
        ))}

        <View style={s.bottomActions}>
           <TouchableOpacity style={s.primaryAction} onPress={() => alert('Report Signed. Syncing with PKH Hospital HIS...')}>
              <Text style={s.primaryActionText}>Approve & Sign Report</Text>
           </TouchableOpacity>
        </View>
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
  backBtn: { width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center' },
  headerTitle: { color: theme.foreground, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold },
  shareBtn: { width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center' },
  scroll: { padding: spacing.lg, gap: spacing.lg, paddingBottom: spacing.xxl },
  
  metaCard: { backgroundColor: theme.card, borderRadius: radius.lg, borderLeftWidth: 4, borderLeftColor: theme.destructive, padding: spacing.lg, gap: 10 },
  metaHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  idBadge: { backgroundColor: theme.backgroundDeep, paddingHorizontal: 8, paddingVertical: 4, borderRadius: radius.sm },
  idText: { color: theme.mutedForeground, fontSize: typography.xs, fontWeight: '700', fontFamily: fontFamily.bold },
  statusBadge: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 4, borderRadius: radius.full },
  statusText: { fontSize: typography.xs, fontWeight: '700', textTransform: 'uppercase', fontFamily: fontFamily.bold },
  findingTitle: { color: theme.foreground, fontSize: typography.xl, fontWeight: '700', fontFamily: fontFamily.bold },
  metaRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  metaText: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  dot: { width: 4, height: 4, borderRadius: 2, backgroundColor: theme.mutedForeground, opacity: 0.3 },

  vizPlaceholder: { 
    height: 220, 
    backgroundColor: theme.backgroundDeep, 
    borderRadius: radius.lg, 
    borderWidth: 1, 
    borderColor: theme.cardBorder, 
    alignItems: 'center', 
    justifyContent: 'center',
    padding: spacing.xl
  },
  vizText: { color: theme.foreground, fontWeight: '700', fontSize: typography.base, fontFamily: fontFamily.bold, marginBottom: 4 },
  vizSubtext: { color: theme.mutedForeground, fontSize: typography.xs, textAlign: 'center', fontFamily: fontFamily.regular, marginBottom: spacing.md },
  expandBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: theme.primaryGlow, paddingHorizontal: 12, paddingVertical: 8, borderRadius: radius.md },
  expandText: { color: theme.primary, fontSize: typography.sm, fontWeight: '600', fontFamily: fontFamily.semiBold },

  confidenceCard: { backgroundColor: theme.card, borderRadius: radius.lg, padding: spacing.lg, gap: spacing.sm, borderWidth: 1, borderColor: theme.cardBorder },
  confidenceHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  confidenceLabel: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.medium },
  confidenceValue: { color: theme.primary, fontWeight: '800', fontSize: typography.lg, fontFamily: fontFamily.bold },
  barBg: { height: 8, backgroundColor: theme.backgroundDeep, borderRadius: 4, overflow: 'hidden' },
  barFill: { height: '100%', backgroundColor: theme.primary },

  reportSection: { gap: spacing.sm },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  sectionTitle: { color: theme.foreground, fontWeight: '700', fontSize: typography.base, fontFamily: fontFamily.bold },
  sectionContent: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular, paddingLeft: 28 },

  bottomActions: { marginTop: spacing.md },
  primaryAction: { backgroundColor: theme.primary, height: 56, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center' },
  primaryActionText: { color: theme.primaryForeground, fontWeight: '700', fontSize: typography.base, fontFamily: fontFamily.bold },
});
