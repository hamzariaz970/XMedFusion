import React, { useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, StatusBar } from 'react-native';
import { Activity, ShieldCheck, Sparkles } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { createShadow, glassPanel, mutedCard, shellBackground } from '../theme/ui';

export default function SplashScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, { toValue: 1, duration: 1000, useNativeDriver: true }).start();
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.16, duration: 900, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 900, useNativeDriver: true }),
      ])
    ).start();
  }, [fadeAnim, pulseAnim]);

  const s = styles(theme);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <Animated.View style={[s.content, { opacity: fadeAnim }]}>
        <View style={s.heroCard}>
          <View style={s.logoRow}>
            <Animated.View style={[s.logoBox, { transform: [{ scale: pulseAnim }] }]}>
              <Activity color={theme.primaryForeground} size={32} />
            </Animated.View>
            <View>
              <Text style={s.brandName}>
                XMed<Text style={s.brandAccent}>Fusion</Text>
              </Text>
              <Text style={s.tagline}>Evidence-grounded radiology AI</Text>
            </View>
          </View>
          <Text style={s.heroCopy}>
            Secure report generation, explainability, and knowledge graph reasoning for modern clinical workflows.
          </Text>
          <View style={s.healthGrid}>
            {[{ label: 'Backend API', ok: true }, { label: 'Inference', ok: true }, { label: 'Knowledge Graph', ok: true }].map((item) => (
              <View key={item.label} style={s.healthChip}>
                <Text style={[s.healthDot, { color: item.ok ? theme.success : theme.destructive }]}>•</Text>
                <Text style={s.healthLabel}>{item.label}</Text>
              </View>
            ))}
          </View>
          <View style={s.statusRow}>
            <ShieldCheck color={theme.primary} size={15} />
            <Text style={s.statusText}>Secure AI engine initializing</Text>
          </View>
        </View>
      </Animated.View>
      <Animated.View style={[s.ctaContainer, { opacity: fadeAnim }]}>
        <TouchableOpacity style={s.ctaButton} onPress={() => navigation.navigate('Login')} activeOpacity={0.9}>
          <Sparkles color={theme.primaryForeground} size={18} />
          <Text style={s.ctaText}>Open Workspace</Text>
        </TouchableOpacity>
        <Text style={s.versionText}>v1.0.0 · Clinical Build</Text>
      </Animated.View>
    </View>
  );
}

const styles = (theme: ReturnType<typeof useTheme>['theme']) =>
  StyleSheet.create({
    container: {
      flex: 1,
      ...shellBackground(theme),
      alignItems: 'center',
      justifyContent: 'center',
      paddingHorizontal: spacing.lg,
    },
    content: { alignItems: 'center', flex: 1, justifyContent: 'center', width: '100%' },
    heroCard: { ...glassPanel(theme), width: '100%', padding: spacing.xl, gap: spacing.md },
    logoRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
    logoBox: {
      width: 72,
      height: 72,
      borderRadius: radius.lg,
      backgroundColor: theme.primary,
      alignItems: 'center',
      justifyContent: 'center',
      ...createShadow(theme, 'lg'),
    },
    brandName: { fontSize: typography['2xl'], color: theme.foreground, fontFamily: fontFamily.extraBold },
    brandAccent: { color: theme.primary },
    tagline: { fontSize: typography.sm, color: theme.mutedForeground, marginTop: 2, fontFamily: fontFamily.medium },
    heroCopy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
    statusRow: { ...mutedCard(theme), flexDirection: 'row', alignItems: 'center', gap: spacing.sm, alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 8 },
    statusText: { color: theme.foreground, fontSize: typography.xs, fontFamily: fontFamily.semiBold },
    healthGrid: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap' },
    healthChip: { ...mutedCard(theme), flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.sm, paddingVertical: 8, gap: 4 },
    healthDot: { fontSize: 10 },
    healthLabel: { color: theme.foreground, fontSize: typography.xs, fontFamily: fontFamily.medium },
    ctaContainer: { width: '100%', alignItems: 'center', paddingBottom: spacing.xxl },
    ctaButton: { width: '100%', backgroundColor: theme.primary, height: 56, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: 8, ...createShadow(theme, 'lg') },
    ctaText: { color: theme.primaryForeground, fontSize: typography.base, fontFamily: fontFamily.bold },
    versionText: { color: theme.mutedForeground, fontSize: typography.xs, marginTop: spacing.md, fontFamily: fontFamily.regular },
  });
