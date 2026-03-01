import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  ScrollView,
  Image as RNImage,
  ActivityIndicator,
  Alert,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { useNavigation } from '@react-navigation/native';
import { useReportStore } from '../store/reportStore';
import Svg, { Circle, Line, Text as SvgText, G } from 'react-native-svg';
import { calculateLayout, nodeTypeColors } from './KnowledgeGraphScreen';
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
  ChevronLeft,
  ChevronDown,
  Search,
  Activity,
  History,
  AlertTriangle,
  Lightbulb
} from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { uploadXRay, StreamChunk } from '../api/api';

type AnalysisStep = {
  id: string;
  label: string;
  status: 'pending' | 'running' | 'done' | 'error';
  icon: any;
};

type ReportSection = {
  title: string;
  content: string;
  icon: any;
};

const initialSteps: AnalysisStep[] = [
  { id: 'validating', label: 'Image Validation', status: 'pending', icon: ShieldCheck },
  { id: 'agents', label: 'Multi-Agent Analysis', status: 'pending', icon: Zap },
  { id: 'synthesis', label: 'Report Synthesis', status: 'pending', icon: FileText },
  { id: 'verify', label: 'Final Verification', status: 'pending', icon: CheckCircle2 },
];

export default function UploadAnalysisScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const [phase, setPhase] = useState<'idle' | 'analyzing' | 'done'>('idle');
  const [steps, setSteps] = useState<AnalysisStep[]>(initialSteps);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [report, setReport] = useState<ReportSection[]>([]);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [explainability, setExplainability] = useState<any>(null);
  const [expandedSection, setExpandedSection] = useState<string | null>('findings');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [rawKgData, setRawKgData] = useState<any>(null);

  const addReport = useReportStore((state) => state.addReport);

  const exportPdf = async () => {
    if (report.length === 0) return;
    try {

      let svgHtml = '';
      if (rawKgData && rawKgData.entities) {
        const width = 600;
        const height = 600;
        const nodes = calculateLayout(rawKgData.entities, width);
        const links = rawKgData.relations.map(([s, t, l]: any) => {
          const src = nodes.find(n => n.id === s.toString());
          const tgt = nodes.find(n => n.id === t.toString());
          return { src, tgt, label: l };
        });
        const cols = nodeTypeColors({ primary: '#2563eb', white: '#fff', mutedForeground: '#9ca3af' });

        let linksHtml = links.map((l: any) => {
          if (!l.src || !l.tgt) return '';
          return `
                  <line x1="${l.src.x}" y1="${l.src.y}" x2="${l.tgt.x}" y2="${l.tgt.y}" stroke="#cbd5e1" stroke-width="1.5" opacity="0.4" />
                  <text x="${(l.src.x + l.tgt.x) / 2}" y="${(l.src.y + l.tgt.y) / 2 - 5}" fill="#9ca3af" font-size="9" text-anchor="middle" font-family="sans-serif">${l.label}</text>
              `;
        }).join('');

        let nodesHtml = nodes.map(n => {
          const c = (cols as any)[n.type] || cols.uncertain;
          return `
                  <g transform="translate(${n.x}, ${n.y})">
                     <circle r="22" fill="#ffffff" stroke="#e2e8f0" stroke-width="2" />
                     <circle r="16" fill="${c.bg}" opacity="0.2" />
                     <text y="35" fill="#1f2937" font-size="10" font-weight="600" text-anchor="middle" font-family="sans-serif">${n.label.length > 15 ? n.label.substring(0, 15) + '...' : n.label}</text>
                  </g>
              `;
        }).join('');

        svgHtml = `
            <div class="section">
                <div class="section-title">Knowledge Graph Visualization</div>
                <div style="width: 100%; overflow-x: auto; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; text-align: center;">
                   <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
                      ${linksHtml}
                      ${nodesHtml}
                   </svg>
                </div>
            </div>
          `;
      }

      let htmlContent = `
        <html>
          <head>
            <style>
              body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding: 40px; color: #111; }
              h1 { color: #2563eb; font-size: 26px; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
              .section { margin-top: 24px; }
              .section-title { font-size: 18px; font-weight: bold; color: #1e40af; margin-bottom: 8px; text-transform: uppercase; }
              .content { font-size: 15px; line-height: 1.6; color: #374151; white-space: pre-wrap; }
              .footer { margin-top: 50px; font-size: 12px; color: #9ca3af; text-align: center; border-top: 1px solid #e5e7eb; padding-top: 20px; }
            </style>
          </head>
          <body>
            <h1>XMedFusion Medical Report</h1>
            <p><strong>Date Generated:</strong> ${new Date().toLocaleString()}</p>
      `;

      report.forEach(sec => {
        htmlContent += `
           <div class="section">
             <div class="section-title">${sec.title}</div>
             <div class="content">${sec.content}</div>
           </div>
         `;
      });

      htmlContent += svgHtml;

      htmlContent += `
            <div class="footer">Analysis strictly provided by XMedFusion AI. For informational purposes only.</div>
          </body>
        </html>
      `;

      const { uri } = await Print.printToFileAsync({ html: htmlContent });
      await Sharing.shareAsync(uri, { UTI: '.pdf', mimeType: 'application/pdf' });
    } catch (e) {
      Alert.alert('Export Error', 'Failed to generate and share PDF.');
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      quality: 0.8,
    });

    if (!result.canceled) {
      handleUpload(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission denied', 'We need camera access to take photos.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 0.8,
    });

    if (!result.canceled) {
      handleUpload(result.assets[0].uri);
    }
  };

  const handleUpload = async (uri: string) => {
    setImageUri(uri);
    setPhase('analyzing');
    setSteps(initialSteps);
    setErrorMessage(null);
    setReport([]);
    setHeatmap(null);
    setRawKgData(null);

    await uploadXRay(uri, (chunk: StreamChunk) => {
      setSteps(prev => {
        const next = [...prev];
        if (chunk.status === 'validating') {
          next[0].status = 'running';
        } else if (chunk.status === 'parallel_start') {
          next[0].status = 'done';
          next[1].status = 'running';
        } else if (chunk.status === 'parallel_done') {
          next[1].status = 'done';
        } else if (chunk.status === 'synthesis_start') {
          next[2].status = 'running';
        } else if (chunk.status === 'complete') {
          next[2].status = 'done';
          next[3].status = 'done';

          let findingsText = '';
          let impressionText = chunk.final_report || '';
          if (impressionText.includes('FINDINGS:')) {
            findingsText = impressionText.split('FINDINGS:')[1].split('IMPRESSION:')[0].trim();
            impressionText = impressionText.split('IMPRESSION:')[1]?.trim() || '';
          }

          addReport({
            id: Math.random().toString(36).substr(2, 9),
            patientId: 'PT-' + Math.floor(1000 + Math.random() * 9000),
            date: new Date().toISOString(),
            findings: findingsText,
            impression: impressionText,
            knowledgeGraph: chunk.knowledge_graph
          });
          setRawKgData(chunk.knowledge_graph);
        } else if (chunk.status === 'error') {
          const runningIdx = next.findIndex(s => s.status === 'running');
          if (runningIdx !== -1) next[runningIdx].status = 'error';
          setErrorMessage(chunk.message || 'An unknown error occurred.');
        }
        return next;
      });

      if (chunk.final_report) {
        // Parse unstructured or semi-structured report into sections
        const sections: ReportSection[] = [];
        const content = chunk.final_report;

        // Simple logic to extract sections if they exist, or just use one
        if (content.includes('FINDINGS:')) {
          const findings = content.split('FINDINGS:')[1].split('IMPRESSION:')[0].trim();
          const impression = content.split('IMPRESSION:')[1]?.trim() || '';
          sections.push({ title: 'Findings', content: findings, icon: Search });
          sections.push({ title: 'Impression', content: impression, icon: Activity });
        } else {
          sections.push({ title: 'Analysis', content, icon: FileText });
        }

        setReport(sections);
        setPhase('done');
      }

      if (chunk.heatmap) {
        setHeatmap(chunk.heatmap);
      }

      if (chunk.explainability) {
        setExplainability(chunk.explainability);
      }
    });
  };

  const reset = () => {
    setPhase('idle');
    setSteps(initialSteps);
    setImageUri(null);
    setReport([]);
    setHeatmap(null);
    setErrorMessage(null);
    setExplainability(null);
    setRawKgData(null);
  };

  const stepStatusColor = (status: AnalysisStep['status']) => {
    if (status === 'done') return theme.success;
    if (status === 'running') return theme.primary;
    if (status === 'error') return theme.destructive;
    return theme.mutedForeground;
  };

  const s = styles(theme);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={theme.backgroundDeep} />
      <View style={s.header}>
        <TouchableOpacity style={s.backBtn} onPress={() => navigation.goBack()}>
          <ChevronLeft color={theme.foreground} size={28} />
        </TouchableOpacity>
        <View>
          <Text style={s.title}>X-Ray Analysis</Text>
          <Text style={s.subtitle}>AI-powered radiology report generation</Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        {/* Upload Area */}
        {phase === 'idle' && (
          <>
            <TouchableOpacity style={s.uploadBox} activeOpacity={0.75} onPress={pickImage}>
              <ImagePlus color={theme.primary} size={48} strokeWidth={1.5} />
              <Text style={s.uploadTitle}>Tap to Select X-Ray</Text>
              <Text style={s.uploadSubtitle}>JPEG, PNG or DICOM · Max 20MB</Text>
            </TouchableOpacity>

            <View style={s.orRow}>
              <View style={s.orLine} />
              <Text style={s.orText}>or</Text>
              <View style={s.orLine} />
            </View>

            <TouchableOpacity style={s.cameraButton} onPress={takePhoto}>
              <Camera color={theme.foreground} size={20} style={{ marginRight: 8 }} />
              <Text style={s.cameraButtonText}>Capture with Camera</Text>
            </TouchableOpacity>
          </>
        )}

        {/* Analysis In Progress */}
        {(phase === 'analyzing' || phase === 'done') && (
          <>
            {/* X-Ray Image */}
            <View style={s.imageContainer}>
              {heatmap ? (
                <RNImage source={{ uri: heatmap }} style={s.xrayImage} resizeMode="contain" />
              ) : imageUri ? (
                <RNImage source={{ uri: imageUri }} style={s.xrayImage} resizeMode="contain" />
              ) : (
                <View style={s.xrayPlaceholder}>
                  <ActivityIndicator color={theme.primary} />
                </View>
              )}
              {heatmap && (
                <View style={s.heatmapBadge}>
                  <Lightbulb color={theme.primary} size={12} />
                  <Text style={s.heatmapBadgeText}>AI Heatmap Active</Text>
                </View>
              )}
            </View>

            {/* Error Message */}
            {errorMessage && (
              <View style={s.errorCard}>
                <AlertTriangle color={theme.destructive} size={20} />
                <View style={{ flex: 1 }}>
                  <Text style={s.errorText}>{errorMessage}</Text>
                  <TouchableOpacity onPress={reset} style={s.retryBtn}>
                    <Text style={s.retryBtnText}>Try Again</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}

            {/* Agent Steps */}
            {!errorMessage && (
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
                      {step.status === 'error' && <AlertTriangle color={theme.backgroundDeep} size={16} />}
                      {step.status === 'pending' && <View style={s.pendingInner} />}
                    </View>
                    <View style={s.stepIconWrapper}>
                      <step.icon color={step.status === 'pending' ? theme.mutedForeground : theme.foreground} size={16} />
                    </View>
                    <Text style={s.stepLabel}>{step.label}</Text>
                    <Text style={[s.stepStatus, { color: stepStatusColor(step.status) }]}>
                      {step.status === 'done' ? 'Done' : step.status === 'running' ? 'Running' : step.status === 'error' ? 'Failed' : '—'}
                    </Text>
                  </View>
                ))}
              </View>
            )}

            {/* Report Sections */}
            {phase === 'done' && report.length > 0 && (
              <>
                <Text style={s.reportTitle}>Generated Report</Text>
                {report.map((section) => (
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

                {/* Reasoning Trace */}
                {explainability && (
                  <View style={s.reasoningCard}>
                    <Text style={s.reasoningTitle}>AI Reasoning Trace</Text>
                    {explainability.reasoning_steps?.map((step: string, i: number) => (
                      <Text key={i} style={s.reasoningStep}>{step}</Text>
                    ))}
                  </View>
                )}

                {/* Knowledge Graph Inline View */}
                {rawKgData && rawKgData.entities && (() => {
                  const width = 600;
                  const height = 500;
                  const nodes = calculateLayout(rawKgData.entities, width);
                  const links = rawKgData.relations.map(([s, t, l]: any) => ({
                    source: Math.max(0, Math.min(s, nodes.length - 1)).toString(),
                    target: Math.max(0, Math.min(t, nodes.length - 1)).toString(),
                    label: l
                  }));
                  const cols = nodeTypeColors(theme);
                  return (
                    <View style={s.kgContainer}>
                      <Text style={s.kgTitle}>Knowledge Graph</Text>
                      <View style={s.kgCanvasWrapper}>
                        <ScrollView horizontal showsHorizontalScrollIndicator={true}>
                          <Svg width={width + 50} height={height} viewBox={`0 0 ${width + 50} ${height}`}>
                            {links.map((link: any, idx: number) => {
                              const srcNode = nodes.find(n => n.id === link.source);
                              const tgtNode = nodes.find(n => n.id === link.target);
                              if (!srcNode || !tgtNode) return null;
                              return (
                                <G key={`link-${idx}`}>
                                  <Line
                                    x1={srcNode.x} y1={srcNode.y}
                                    x2={tgtNode.x} y2={tgtNode.y}
                                    stroke={theme.border}
                                    strokeWidth={1.5}
                                    opacity={0.3}
                                  />
                                  <SvgText
                                    x={(srcNode.x + tgtNode.x) / 2}
                                    y={(srcNode.y + tgtNode.y) / 2 - 5}
                                    fill={theme.mutedForeground}
                                    fontSize={9}
                                    textAnchor="middle"
                                    fontFamily={fontFamily.regular}
                                  >
                                    {link.label}
                                  </SvgText>
                                </G>
                              );
                            })}
                            {nodes.map((node: any) => {
                              const tColor = (cols as any)[node.type] || cols.uncertain;
                              return (
                                <G key={`node-${node.id}`} x={node.x} y={node.y}>
                                  <Circle r={22} fill={theme.card} stroke={theme.border} strokeWidth={2} />
                                  <Circle r={14} fill={tColor.fill} />
                                  <SvgText
                                    y={36}
                                    fill={theme.foreground}
                                    fontSize={10}
                                    fontWeight="600"
                                    textAnchor="middle"
                                    fontFamily={fontFamily.semiBold}
                                  >
                                    {node.label.length > 15 ? node.label.substring(0, 15) + '...' : node.label}
                                  </SvgText>
                                </G>
                              );
                            })}
                          </Svg>
                        </ScrollView>
                      </View>
                    </View>
                  );
                })()}

                {/* Actions */}
                <View style={s.actionRow}>
                  <TouchableOpacity style={s.actionBtn} onPress={exportPdf}>
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
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  backBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.cardBorder,
    alignItems: 'center',
    justifyContent: 'center',
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
  imageContainer: {
    height: 220,
    backgroundColor: theme.backgroundDeep,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: theme.cardBorder,
    overflow: 'hidden',
    position: 'relative',
    alignItems: 'center',
    justifyContent: 'center',
  },
  xrayImage: { width: '100%', height: '100%' },
  xrayPlaceholder: { alignItems: 'center', justifyContent: 'center' },
  heatmapBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radius.full,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  heatmapBadgeText: { color: theme.primary, fontSize: 10, fontWeight: '600', fontFamily: fontFamily.semiBold },
  errorCard: {
    backgroundColor: theme.destructiveBg,
    borderWidth: 1,
    borderColor: theme.destructive,
    borderRadius: radius.lg,
    padding: spacing.md,
    flexDirection: 'row',
    gap: spacing.md,
  },
  errorText: { color: theme.destructive, fontSize: typography.sm, fontFamily: fontFamily.medium },
  retryBtn: { marginTop: spacing.sm },
  retryBtnText: { color: theme.foreground, fontSize: typography.sm, fontWeight: '700', textDecorationLine: 'underline' },
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
  reasoningCard: {
    backgroundColor: theme.primaryGlow,
    borderWidth: 1,
    borderColor: theme.primary,
    borderRadius: radius.lg,
    padding: spacing.md,
    gap: spacing.sm,
  },
  reasoningTitle: { color: theme.primary, fontWeight: '700', fontSize: typography.sm, fontFamily: fontFamily.bold },
  reasoningStep: { color: theme.foreground, fontSize: typography.xs, opacity: 0.9, lineHeight: 16, fontFamily: fontFamily.regular },
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
  kgContainer: {
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.cardBorder,
    borderRadius: radius.lg,
    padding: spacing.md,
    gap: spacing.sm,
  },
  kgTitle: { color: theme.foreground, fontWeight: '700', fontSize: typography.base, fontFamily: fontFamily.bold, marginBottom: spacing.xs },
  kgCanvasWrapper: { height: 350, backgroundColor: theme.backgroundDeep, borderRadius: radius.md, borderWidth: 1, borderColor: theme.border, overflow: 'hidden' },
});
