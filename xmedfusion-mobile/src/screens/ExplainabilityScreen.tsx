import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, StatusBar, Image } from 'react-native';
import { Activity, ArrowLeft, Brain, FileSearch, Info, Sparkles } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { glassPanel, shellBackground, surfaceCard } from '../theme/ui';
import { useAnalysis } from '../context/AnalysisContext';

export default function ExplainabilityScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const { previewUrl, referenceImageUrl, heatmapData, report, detectedModality, explainabilityData } = useAnalysis();
  const s = styles(theme);

  const displayReferenceImage = referenceImageUrl || previewUrl;
  const modalityLabel = detectedModality === 'ct' ? 'CT scan' : 'X-ray';
  const inputPanelLabel = detectedModality === 'ct' ? 'Model Input Montage' : 'Original Scan';

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView contentInsetAdjustmentBehavior="automatic" contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.heroCard}>
          <View style={s.badge}>
            <Sparkles color={theme.primary} size={14} />
            <Text style={s.badgeText}>AI Interpretability</Text>
          </View>
          <Text style={s.title}>Explainability Module</Text>
          <Text style={s.copy}>Visualize the evidence behind generated reports with categorical anatomical highlights and automated diagnostic reasoning.</Text>
          <TouchableOpacity style={s.backButton} onPress={() => navigation.navigate('/upload')}>
            <ArrowLeft color={theme.foreground} size={16} />
            <Text style={s.backText}>Back to Upload</Text>
          </TouchableOpacity>
        </View>

        {!displayReferenceImage ? (
          <View style={s.emptyCard}>
            <FileSearch color={theme.mutedForeground} size={32} />
            <Text style={s.emptyTitle}>No image data found</Text>
            <Text style={s.emptyCopy}>Please upload a scan to begin analysis.</Text>
          </View>
        ) : (
          <View style={s.grid}>
            <View style={s.card}>
              <View style={s.cardHeader}>
                <Activity color={theme.primary} size={18} />
                <Text style={s.cardTitle}>{inputPanelLabel}</Text>
              </View>
              <View style={s.imageBox}>
                <Image source={{ uri: displayReferenceImage }} style={s.image} resizeMode="contain" />
              </View>
              <Text style={s.caption}>{modalityLabel}</Text>
            </View>

            <View style={s.card}>
              <View style={s.cardHeader}>
                <Info color={theme.primary} size={18} />
                <Text style={s.cardTitle}>AI-Annotated Scan</Text>
              </View>
              <View style={s.imageBox}>
                {heatmapData ? (
                  <Image source={{ uri: heatmapData }} style={s.image} resizeMode="contain" />
                ) : (
                  <Text style={s.annotatedText}>No visual highlights generated for this scan.</Text>
                )}
              </View>
            </View>

            <View style={[s.card, { width: '100%' }]}>
              <View style={s.cardHeader}>
                <Brain color={theme.primary} size={18} />
                <Text style={s.cardTitle}>Automated Diagnostic Narrative</Text>
              </View>
              <Text style={s.sectionLabel}>Findings</Text>
              <Text style={s.sectionBody}>{report?.findings || 'No findings recorded.'}</Text>
              <Text style={s.sectionLabel}>Impression</Text>
              <Text style={s.sectionBody}>{report?.impression || 'No impression recorded.'}</Text>
              {explainabilityData?.reasoning_steps?.length ? (
                <>
                  <Text style={s.sectionLabel}>Reasoning Trace</Text>
                  {explainabilityData.reasoning_steps.map((step: string, index: number) => (
                    <Text key={`${step}-${index}`} style={s.reasoningStep}>{step}</Text>
                  ))}
                </>
              ) : null}
            </View>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) =>
  StyleSheet.create({
    container: { flex: 1, ...shellBackground(theme) },
    scroll: { paddingHorizontal: spacing.lg, paddingTop: spacing.xl, paddingBottom: spacing.xxl + 84, gap: spacing.md },
    heroCard: { ...glassPanel(theme), padding: spacing.lg, gap: spacing.md },
    badge: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 6, borderRadius: radius.full, backgroundColor: theme.primaryGlow },
    badgeText: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
    title: { color: theme.foreground, fontSize: typography['2xl'], fontFamily: fontFamily.extraBold },
    copy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
    backButton: { flexDirection: 'row', alignItems: 'center', gap: 8, alignSelf: 'flex-start', borderWidth: 1, borderColor: theme.cardBorder, backgroundColor: theme.card, paddingHorizontal: spacing.md, paddingVertical: 10, borderRadius: radius.lg },
    backText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
    emptyCard: { ...glassPanel(theme), padding: spacing.lg, alignItems: 'center', gap: spacing.sm },
    emptyTitle: { color: theme.foreground, fontSize: typography.lg, fontFamily: fontFamily.bold },
    emptyCopy: { color: theme.mutedForeground, textAlign: 'center', lineHeight: 22, fontFamily: fontFamily.regular },
    grid: { gap: spacing.md },
    card: { ...surfaceCard(theme), padding: spacing.md, gap: spacing.sm },
    cardHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    cardTitle: { color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.bold },
    imageBox: { minHeight: 220, borderRadius: radius.lg, backgroundColor: theme.backgroundDeep, alignItems: 'center', justifyContent: 'center', overflow: 'hidden' },
    image: { width: '100%', height: 240 },
    caption: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
    annotatedText: { color: theme.mutedForeground, fontSize: typography.sm, textAlign: 'center', padding: spacing.lg, fontFamily: fontFamily.regular },
    sectionLabel: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.bold, marginTop: spacing.xs },
    sectionBody: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
    reasoningStep: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 20, fontFamily: fontFamily.regular },
  });
