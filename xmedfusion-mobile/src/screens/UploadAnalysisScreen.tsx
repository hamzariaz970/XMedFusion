import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  ScrollView,
  Image,
  ActivityIndicator,
} from 'react-native';
import { 
  ImagePlus, 
  Camera, 
  Zap, 
  FileText, 
  CheckCircle2, 
  ShieldCheck,
  Share,
  RotateCcw,
  ChevronUp,
  ChevronDown,
  Info,
  Scan,
  Search,
  Activity,
  History
} from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

type AnalysisStep = {
  id: string;
  label: string;
  status: 'pending' | 'running' | 'done';
  icon: any;
};

type ReportSection = {
  title: string;
  content: string;
  icon: any;
};

const initialSteps: AnalysisStep[] = [
  { id: 'visual', label: 'Visual Feature Extraction', status: 'pending', icon: Scan },
  { id: 'rag', label: 'RAG Knowledge Retrieval', status: 'pending', icon: History },
  { id: 'report', label: 'Report Generation (LLM)', status: 'pending', icon: FileText },
  { id: 'verify', label: 'Clinical Verification', status: 'pending', icon: ShieldCheck },
];

export default function UploadAnalysisScreen() {
  const { theme, isDark } = useTheme();
  const [phase, setPhase] = useState<'idle' | 'analyzing' | 'done'>('idle');
  const [steps, setSteps] = useState<AnalysisStep[]>(initialSteps);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [expandedSection, setExpandedSection] = useState<string | null>('findings');

  const mockReport: ReportSection[] = [
    {
      title: 'Findings',
      icon: Search,
      content:
        'Bilateral perihilar infiltrates are noted with increased opacity in the right lower lobe consistent with consolidation. The cardiomediastinal silhouette is mildly enlarged. No pneumothorax or significant pleural effusion identified.',
    },
    {
      title: 'Impression',
      icon: Activity,
      content:
        'Findings are consistent with bilateral pneumonia with early consolidative changes in the right lower lobe. Mild cardiomegaly noted. Clinical correlation recommended.',
    },
    {
      title: 'Recommendations',
      icon: ShieldCheck,
      content:
        '1. Start empirical antibiotic therapy.\n2. Repeat chest X-ray in 48-72 hours to assess response.\n3. Echocardiogram to evaluate cardiac function.\n4. Blood cultures and CBC with differential.',
    },
  ];

  const simulateAnalysis = () => {
    setPhase('analyzing');
    const updatedSteps = [...initialSteps];

    // Step 1
    setTimeout(() => {
      updatedSteps[0] = { ...updatedSteps[0], status: 'running' };
      setSteps([...updatedSteps]);
    }, 500);
    setTimeout(() => {
      updatedSteps[0] = { ...updatedSteps[0], status: 'done' };
      updatedSteps[1] = { ...updatedSteps[1], status: 'running' };
      setSteps([...updatedSteps]);
    }, 2000);
    setTimeout(() => {
      updatedSteps[1] = { ...updatedSteps[1], status: 'done' };
      updatedSteps[2] = { ...updatedSteps[2], status: 'running' };
      setSteps([...updatedSteps]);
    }, 3800);
    setTimeout(() => {
      updatedSteps[2] = { ...updatedSteps[2], status: 'done' };
      updatedSteps[3] = { ...updatedSteps[3], status: 'running' };
      setSteps([...updatedSteps]);
    }, 6000);
    setTimeout(() => {
      updatedSteps[3] = { ...updatedSteps[3], status: 'done' };
      setSteps([...updatedSteps]);
      setPhase('done');
    }, 7500);
  };

  const reset = () => {
    setPhase('idle');
    setSteps(initialSteps);
    setImageUri(null);
  };

  const stepStatusColor = (status: AnalysisStep['status']) => {
    if (status === 'done') return theme.success;
    if (status === 'running') return theme.primary;
    return theme.mutedForeground;
  };

  const s = styles(theme);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={theme.backgroundDeep} />
      <View style={s.header}>
        <Text style={s.title}>X-Ray Analysis</Text>
        <Text style={s.subtitle}>AI-powered radiology report generation</Text>
      </View>

      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        {/* Upload Area */}
        {phase === 'idle' && (
          <>
            <TouchableOpacity style={s.uploadBox} activeOpacity={0.75} onPress={simulateAnalysis}>
              <ImagePlus color={theme.primary} size={48} strokeWidth={1.5} />
              <Text style={s.uploadTitle}>Tap to Select X-Ray</Text>
              <Text style={s.uploadSubtitle}>JPEG, PNG or DICOM · Max 20MB</Text>
            </TouchableOpacity>

            <View style={s.orRow}>
              <View style={s.orLine} />
              <Text style={s.orText}>or</Text>
              <View style={s.orLine} />
            </View>

            <TouchableOpacity style={s.cameraButton} onPress={simulateAnalysis}>
               <Camera color={theme.foreground} size={20} style={{ marginRight: 8 }} />
               <Text style={s.cameraButtonText}>Capture with Camera</Text>
            </TouchableOpacity>

            {/* Demo note */}
            <View style={s.demoNote}>
              <Info color={theme.primary} size={16} style={{ marginTop: 2 }} />
              <Text style={s.demoNoteText}>
                Tap either option above to preview the AI analysis flow with a demo report.
              </Text>
            </View>
          </>
        )}

        {/* Analysis In Progress */}
        {(phase === 'analyzing' || phase === 'done') && (
          <>
            {/* X-Ray placeholder */}
            <View style={s.xrayPlaceholder}>
              <Scan color={theme.mutedForeground} size={48} strokeWidth={1} />
              <Text style={s.xrayPlaceholderText}>Chest_PA_2847.jpg</Text>
            </View>

            {/* Agent Steps */}
            <View style={s.stepsCard}>
              <View style={s.stepsHeader}>
                {phase === 'done' ? (
                   <CheckCircle2 color={theme.success} size={20} />
                ) : (
                   <Zap color={theme.primary} size={20} />
                )}
                <Text style={s.stepsTitle}>
                  {phase === 'done' ? 'Analysis Complete' : 'Agentic Pipeline Running...'}
                </Text>
              </View>
              {steps.map((step) => (
                <View key={step.id} style={s.stepRow}>
                  <View style={[s.stepDot, { backgroundColor: stepStatusColor(step.status) }]}>
                    {step.status === 'running' && (
                      <ActivityIndicator size="small" color={theme.backgroundDeep} />
                    )}
                    {step.status === 'done' && <CheckCircle2 color={theme.backgroundDeep} size={16} strokeWidth={3} />}
                    {step.status === 'pending' && <View style={s.pendingInner} />}
                  </View>
                  <View style={s.stepIconWrapper}>
                     <step.icon color={step.status === 'pending' ? theme.mutedForeground : theme.foreground} size={16} />
                  </View>
                  <Text style={s.stepLabel}>{step.label}</Text>
                  <Text style={[s.stepStatus, { color: stepStatusColor(step.status) }]}>
                    {step.status === 'done' ? 'Done' : step.status === 'running' ? 'Running' : '—'}
                  </Text>
                </View>
              ))}
            </View>

            {/* Report Sections */}
            {phase === 'done' && (
              <>
                <Text style={s.reportTitle}>Generated Report</Text>
                {mockReport.map((section) => (
                  <TouchableOpacity
                    key={section.title}
                    style={s.reportSection}
                    onPress={() =>
                      setExpandedSection(expandedSection === section.title ? null : section.title)
                    }
                    activeOpacity={0.8}
                  >
                    <View style={s.reportSectionHeader}>
                      <View style={s.reportSectionTitleRow}>
                        <section.icon color={theme.foreground} size={18} style={{ marginRight: 8 }} />
                        <Text style={s.reportSectionTitle}>{section.title}</Text>
                      </View>
                      {expandedSection === section.title ? (
                         <ChevronUp color={theme.mutedForeground} size={18} />
                      ) : (
                         <ChevronDown color={theme.mutedForeground} size={18} />
                      )}
                    </View>
                    {expandedSection === section.title && (
                      <Text style={s.reportContent}>{section.content}</Text>
                    )}
                  </TouchableOpacity>
                ))}

                {/* Actions */}
                <View style={s.actionRow}>
                  <TouchableOpacity style={s.actionBtn}>
                    <Share color={theme.primaryForeground} size={18} style={{ marginRight: 8 }} />
                    <Text style={s.actionBtnText}>Export PDF</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={[s.actionBtn, s.actionBtnSecondary]} onPress={reset}>
                    <RotateCcw color={theme.foreground} size={18} style={{ marginRight: 8 }} />
                    <Text style={s.actionBtnSecondaryText}>New Analysis</Text>
                  </TouchableOpacity>
                </View>
              </>
            )}
          </>
        )}
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.background },
  header: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.xxl,
    paddingBottom: spacing.md,
  },
  title: { color: theme.foreground, fontSize: typography['2xl'], fontWeight: '700', fontFamily: fontFamily.bold },
  subtitle: { color: theme.mutedForeground, fontSize: typography.sm, marginTop: 2, fontFamily: fontFamily.regular },
  scroll: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl, gap: spacing.md },
  uploadBox: {
    borderWidth: 2,
    borderColor: theme.cardBorder,
    borderStyle: 'dashed',
    borderRadius: radius.xl,
    backgroundColor: theme.card,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.xxl + spacing.lg,
    gap: spacing.sm,
  },
  uploadIcon: { fontSize: 48 },
  uploadTitle: { color: theme.foreground, fontWeight: '600', fontSize: typography.lg, fontFamily: fontFamily.bold },
  uploadSubtitle: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.regular },
  orRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  orLine: { flex: 1, height: 1, backgroundColor: theme.cardBorder },
  orText: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.medium },
  cameraButton: {
    flexDirection: 'row',
    height: 52,
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.cardBorder,
    borderRadius: radius.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cameraButtonText: { color: theme.foreground, fontWeight: '600', fontSize: typography.base, fontFamily: fontFamily.bold },
  demoNote: {
    flexDirection: 'row',
    backgroundColor: theme.primaryGlow,
    borderWidth: 1,
    borderColor: theme.primary,
    borderRadius: radius.md,
    padding: spacing.md,
    gap: spacing.sm,
  },
  demoNoteText: { flex: 1, color: theme.primary, fontSize: typography.xs, lineHeight: 18, fontFamily: fontFamily.regular },
  xrayPlaceholder: {
    height: 180,
    backgroundColor: theme.backgroundDeep,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: theme.cardBorder,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  xrayPlaceholderText: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.medium },
  stepsCard: {
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.cardBorder,
    borderRadius: radius.lg,
    padding: spacing.lg,
    gap: spacing.md,
  },
  stepsHeader: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.xs },
  stepsTitle: { color: theme.foreground, fontWeight: '700', fontSize: typography.base, fontFamily: fontFamily.bold },
  stepRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  stepDot: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pendingInner: { width: 4, height: 4, borderRadius: 2, backgroundColor: 'rgba(255,255,255,0.3)' },
  stepIconWrapper: { width: 20, alignItems: 'center' },
  stepLabel: { flex: 1, color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.medium },
  stepStatus: { fontSize: typography.xs, fontWeight: '600', fontFamily: fontFamily.semiBold },
  reportTitle: { color: theme.foreground, fontWeight: '700', fontSize: typography.lg, fontFamily: fontFamily.bold },
  reportSection: {
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.cardBorder,
    borderRadius: radius.lg,
    padding: spacing.md,
  },
  reportSectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  reportSectionTitleRow: { flexDirection: 'row', alignItems: 'center' },
  reportSectionTitle: { color: theme.foreground, fontWeight: '600', fontSize: typography.base, fontFamily: fontFamily.semiBold },
  reportContent: {
    color: theme.mutedForeground,
    fontSize: typography.sm,
    lineHeight: 22,
    marginTop: spacing.md,
    fontFamily: fontFamily.regular,
  },
  actionRow: { flexDirection: 'row', gap: spacing.sm },
  actionBtn: {
    flex: 1,
    flexDirection: 'row',
    height: 52,
    backgroundColor: theme.primary,
    borderRadius: radius.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionBtnText: { color: theme.primaryForeground, fontWeight: '700', fontSize: typography.sm, fontFamily: fontFamily.bold },
  actionBtnSecondary: {
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.cardBorder,
  },
  actionBtnSecondaryText: { color: theme.foreground, fontWeight: '600', fontSize: typography.sm, fontFamily: fontFamily.semiBold },
});
