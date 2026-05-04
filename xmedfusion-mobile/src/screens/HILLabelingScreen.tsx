import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, ScrollView, TextInput } from 'react-native';
import { ArrowLeft, CheckCircle2, FileText, Save, Send, ShieldCheck } from 'lucide-react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { glassPanel, shellBackground, surfaceCard } from '../theme/ui';

export default function HILLabelingScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const { theme, isDark } = useTheme();
  const taskId = route.params?.taskId || 'demo';
  const [findings, setFindings] = useState('');
  const [impression, setImpression] = useState('');
  const s = styles(theme);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView contentInsetAdjustmentBehavior="automatic" contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.heroCard}>
          <TouchableOpacity style={s.backButton} onPress={() => navigation.navigate('/patients')}>
            <ArrowLeft color={theme.foreground} size={16} />
            <Text style={s.backText}>Back to Patients</Text>
          </TouchableOpacity>
          <View style={s.badge}>
            <ShieldCheck color={theme.primary} size={14} />
            <Text style={s.badgeText}>Human-in-the-loop review</Text>
          </View>
          <Text style={s.title}>Task {taskId}</Text>
          <Text style={s.copy}>Fill the report fields below and submit the scan for admin review, matching the web app's HIL workflow.</Text>
        </View>

        <View style={s.scanCard}>
          <Text style={s.sectionTitle}>Scan Preview</Text>
          <View style={s.previewBox}>
            <Text style={s.previewText}>Scan image unavailable in this demo route</Text>
          </View>
        </View>

        <View style={s.formCard}>
          <Text style={s.sectionTitle}>Radiology Report</Text>
          <View style={s.field}>
            <Text style={s.label}>Findings</Text>
            <TextInput value={findings} onChangeText={setFindings} placeholder="Describe observations..." placeholderTextColor={theme.mutedForeground} multiline style={s.input} />
          </View>
          <View style={s.field}>
            <Text style={s.label}>Impression</Text>
            <TextInput value={impression} onChangeText={setImpression} placeholder="Summarize your impression..." placeholderTextColor={theme.mutedForeground} multiline style={s.input} />
          </View>
          <View style={s.actions}>
            <TouchableOpacity style={s.secondaryButton}>
              <Save color={theme.foreground} size={16} />
              <Text style={s.secondaryButtonText}>Save Draft</Text>
            </TouchableOpacity>
            <TouchableOpacity style={s.primaryButton}>
              <Send color={theme.primaryForeground} size={16} />
              <Text style={s.primaryButtonText}>Submit Report</Text>
            </TouchableOpacity>
          </View>
          <View style={s.completeRow}>
            <CheckCircle2 color={theme.success} size={16} />
            <Text style={s.completeText}>Submitting routes the scan to admin approval.</Text>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) =>
  StyleSheet.create({
    container: { flex: 1, ...shellBackground(theme) },
    scroll: { paddingHorizontal: spacing.lg, paddingTop: spacing.xl, paddingBottom: spacing.xxl + 84, gap: spacing.md },
    heroCard: { ...glassPanel(theme), padding: spacing.lg, gap: spacing.sm },
    backButton: { flexDirection: 'row', alignItems: 'center', gap: 8, alignSelf: 'flex-start', borderWidth: 1, borderColor: theme.cardBorder, backgroundColor: theme.card, paddingHorizontal: spacing.md, paddingVertical: 10, borderRadius: radius.lg },
    backText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
    badge: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 6, borderRadius: radius.full, backgroundColor: theme.primaryGlow },
    badgeText: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
    title: { color: theme.foreground, fontSize: typography['2xl'], fontFamily: fontFamily.extraBold },
    copy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
    scanCard: { ...surfaceCard(theme), padding: spacing.md, gap: spacing.sm },
    formCard: { ...surfaceCard(theme), padding: spacing.md, gap: spacing.md },
    sectionTitle: { color: theme.foreground, fontSize: typography.lg, fontFamily: fontFamily.bold },
    previewBox: { minHeight: 180, borderRadius: radius.lg, backgroundColor: theme.clinicalInk, alignItems: 'center', justifyContent: 'center', padding: spacing.lg },
    previewText: { color: theme.mutedForeground, textAlign: 'center', fontFamily: fontFamily.regular },
    field: { gap: 6 },
    label: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
    input: { minHeight: 96, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.md, backgroundColor: theme.card, color: theme.foreground, padding: spacing.md, textAlignVertical: 'top', fontFamily: fontFamily.regular },
    actions: { flexDirection: 'row', gap: spacing.sm },
    primaryButton: { flex: 1, backgroundColor: theme.primary, borderRadius: radius.lg, height: 52, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
    primaryButtonText: { color: theme.primaryForeground, fontSize: typography.sm, fontFamily: fontFamily.bold },
    secondaryButton: { flex: 1, borderWidth: 1, borderColor: theme.cardBorder, backgroundColor: theme.card, borderRadius: radius.lg, height: 52, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
    secondaryButtonText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
    completeRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    completeText: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  });
