import React, { useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, ScrollView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Activity, ArrowRight, Brain, CheckCircle2, FileText, Network, ScanSearch, ShieldCheck, Sparkles, Stethoscope } from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { glassPanel, mutedCard, shellBackground, surfaceCard } from '../theme/ui';
import { supabase } from '../lib/supabase';

const headlineMetrics = [
  { label: 'BLEU-1', baseline: '0.0493', value: '0.3359' },
  { label: 'ROUGE-L', baseline: '0.0863', value: '0.2440' },
  { label: 'Consistency', baseline: '2.38', value: '7.80' },
  { label: 'Accuracy', baseline: '2.34', value: '6.93' },
];

const strengths = [
  'Structured perception replaces single-pass generation with explicit intermediate evidence.',
  'Knowledge graph control reduces unsupported diagnostic statements and cross-modal inconsistency.',
  'Evidence-prioritized synthesis resolves conflicts across agents before the final report is produced.',
  'Explainability overlays and traceable findings let radiologists verify what the system actually saw.',
];

export default function IndexScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const [session, setSession] = useState<any>(null);
  const s = styles(theme);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session: nextSession } }) => {
      setSession(nextSession);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
    });

    return () => subscription.unsubscribe();
  }, []);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView contentInsetAdjustmentBehavior="automatic" contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.heroCard}>
          <View style={s.badge}>
            <ShieldCheck color={theme.primaryForeground} size={14} />
            <Text style={s.badgeText}>NUST Final Year Project</Text>
          </View>

          <Text style={s.title}>XMedFusion brings transparent, evidence-grounded AI to radiology report generation.</Text>
          <Text style={s.copy}>
            Our multi-agent framework decomposes chest X-ray reporting into visual perception, knowledge graph construction, retrieval-guided drafting, and evidence-prioritized synthesis to reduce hallucinations and improve diagnostic reliability.
          </Text>

          <View style={s.heroPills}>
            <View style={s.heroPill}><Text style={s.heroPillText}>IU X-ray benchmark</Text></View>
            <View style={s.heroPill}><Text style={s.heroPillText}>BioMedCLIP + MedGemma 1.5:4B</Text></View>
            <View style={s.heroPill}><Text style={s.heroPillText}>Explainable report generation</Text></View>
          </View>

          <View style={s.actions}>
            <TouchableOpacity style={s.primaryButton} onPress={() => navigation.navigate(session ? '/upload' : '/login')}>
              <Text style={s.primaryButtonText}>Upload Scan</Text>
              <ArrowRight color={theme.primaryForeground} size={18} />
            </TouchableOpacity>
            <TouchableOpacity style={s.secondaryButton} onPress={() => navigation.navigate(session ? '/dashboard' : '/login')}>
              <Text style={s.secondaryButtonText}>Open Workspace</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={s.heroCardsRow}>
          {[
            { title: 'Transparent', desc: 'Agent outputs stay reviewable', icon: ShieldCheck },
            { title: 'Multi-Agent', desc: 'Decomposed clinical reasoning', icon: Network },
            { title: 'Grounded', desc: 'No black-box generation', icon: CheckCircle2 },
          ].map((item) => (
            <View key={item.title} style={s.smallHeroCard}>
              <View style={s.cardIcon}>
                <item.icon color={theme.primary} size={22} />
              </View>
              <Text style={s.smallHeroTitle}>{item.title}</Text>
              <Text style={s.smallHeroCopy}>{item.desc}</Text>
            </View>
          ))}
        </View>

        <View style={s.metricsGrid}>
          {headlineMetrics.map((metric, index) => (
            <View key={metric.label} style={[s.metricCard, index % 2 === 0 ? s.metricCardLeft : s.metricCardRight]}>
              <Text style={s.metricLabel}>{metric.label}</Text>
              <View style={s.metricRow}>
                <View>
                  <Text style={s.metricValue}>{metric.value}</Text>
                  <Text style={s.metricSub}>XMedFusion</Text>
                </View>
                <View style={s.metricBaseline}>
                  <Text style={s.metricBaselineValue}>{metric.baseline}</Text>
                  <Text style={s.metricBaselineSub}>LLaVA-Med 1.5</Text>
                </View>
              </View>
            </View>
          ))}
        </View>

        <View style={s.sectionCard}>
          <View style={s.sectionHeader}>
            <Text style={s.sectionEyebrow}>Project Focus</Text>
            <Text style={s.sectionTitle}>Built for faster, more reliable, and more interpretable radiology reporting.</Text>
          </View>
          <Text style={s.sectionCopy}>
            The project targets automated generation of Findings and Impression sections for chest radiographs while preserving traceable evidence at each stage of the pipeline.
          </Text>
        </View>

        <View style={s.grid}>
          {strengths.map((item) => (
            <View key={item} style={s.strengthCard}>
              <Sparkles color={theme.primary} size={18} />
              <Text style={s.strengthText}>{item}</Text>
            </View>
          ))}
        </View>

        <View style={s.agentSection}>
          <Text style={s.sectionEyebrow}>Pipeline Stages</Text>
          <View style={s.grid}>
            {[
              { title: 'Vision Agent', icon: ScanSearch, copy: 'Extracts dense image-grounded descriptions from lung fields, pleura, mediastinum, and cardiac silhouette.' },
              { title: 'KG Agent', icon: Network, copy: 'Builds a RadGraph-compliant knowledge graph so clinical facts stay explicit and checkable.' },
              { title: 'Retrieval & Draft', icon: FileText, copy: 'Retrieves top-k similar cases to supply contextual scaffolding without replacing evidence control.' },
              { title: 'Synthesis Agent', icon: Brain, copy: 'Uses MedGemma 1.5:4B to reconcile vision evidence, graph facts, and retrieved context into a coherent report.' },
            ].map((item) => (
              <View key={item.title} style={s.agentCard}>
                <item.icon color={theme.primary} size={20} />
                <Text style={s.agentTitle}>{item.title}</Text>
                <Text style={s.agentCopy}>{item.copy}</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={s.bottomNote}>
          <Activity color={theme.primary} size={18} />
          <Text style={s.bottomNoteText}>The mobile app now follows the same route structure and visual language as the web app.</Text>
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
    badge: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start', borderWidth: 1, borderColor: theme.primary, backgroundColor: theme.primary, borderRadius: radius.full, paddingHorizontal: spacing.sm, paddingVertical: 6 },
    badgeText: { color: theme.primaryForeground, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
    title: { color: theme.foreground, fontSize: typography['2xl'], fontFamily: fontFamily.extraBold, lineHeight: 34 },
    copy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
    heroPills: { flexDirection: 'row', gap: 6, flexWrap: 'wrap' },
    heroPill: { ...mutedCard(theme), paddingHorizontal: spacing.sm, paddingVertical: 8 },
    heroPillText: { color: theme.foreground, fontSize: typography.xs, fontFamily: fontFamily.semiBold },
    actions: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap' },
    primaryButton: { backgroundColor: theme.primary, borderRadius: radius.full, paddingHorizontal: spacing.md, height: 50, flexDirection: 'row', alignItems: 'center', gap: 8 },
    primaryButtonText: { color: theme.primaryForeground, fontSize: typography.sm, fontFamily: fontFamily.bold },
    secondaryButton: { ...surfaceCard(theme), height: 50, paddingHorizontal: spacing.md, justifyContent: 'center' },
    secondaryButtonText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
    heroCardsRow: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap' },
    smallHeroCard: { ...surfaceCard(theme), width: '48%', padding: spacing.md, gap: 6 },
    cardIcon: { width: 40, height: 40, borderRadius: 12, backgroundColor: theme.primaryGlow, alignItems: 'center', justifyContent: 'center' },
    smallHeroTitle: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.bold },
    smallHeroCopy: { color: theme.mutedForeground, fontSize: typography.xs, lineHeight: 16, fontFamily: fontFamily.regular },
    metricsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
    metricCard: { ...surfaceCard(theme), width: '48%', padding: spacing.md, gap: 8 },
    metricCardLeft: {},
    metricCardRight: {},
    metricLabel: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
    metricRow: { flexDirection: 'row', justifyContent: 'space-between', gap: spacing.sm },
    metricValue: { color: theme.foreground, fontSize: typography.xl, fontFamily: fontFamily.extraBold },
    metricSub: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
    metricBaseline: { alignItems: 'flex-end' },
    metricBaselineValue: { color: theme.mutedForeground, fontSize: typography.base, fontFamily: fontFamily.semiBold },
    metricBaselineSub: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
    sectionCard: { ...glassPanel(theme), padding: spacing.lg, gap: spacing.sm },
    sectionHeader: { gap: 4 },
    sectionEyebrow: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
    sectionTitle: { color: theme.foreground, fontSize: typography.lg, fontFamily: fontFamily.extraBold, lineHeight: 26 },
    sectionCopy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
    grid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
    strengthCard: { ...surfaceCard(theme), width: '48%', padding: spacing.md, gap: 8 },
    strengthText: { color: theme.foreground, fontSize: typography.xs, lineHeight: 18, fontFamily: fontFamily.regular },
    agentSection: { gap: spacing.sm },
    agentCard: { ...surfaceCard(theme), width: '48%', padding: spacing.md, gap: 8 },
    agentTitle: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.bold },
    agentCopy: { color: theme.mutedForeground, fontSize: typography.xs, lineHeight: 16, fontFamily: fontFamily.regular },
    bottomNote: { ...mutedCard(theme), flexDirection: 'row', alignItems: 'center', gap: 8, padding: spacing.md },
    bottomNoteText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold, flex: 1 },
  });
