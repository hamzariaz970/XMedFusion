import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  StatusBar,
} from 'react-native';
import { Activity } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

export default function SplashScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, { toValue: 1, duration: 1000, useNativeDriver: true }).start();
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.3, duration: 800, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    ).start();
  }, []);

  const s = styles(theme);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={theme.backgroundDeep} />
      <Animated.View style={[s.content, { opacity: fadeAnim }]}>
        <View style={s.logoContainer}>
          <View style={s.logoBox}>
            <Activity color={theme.primaryForeground} size={32} />
          </View>
        </View>
        <Text style={s.brandName}>XMed<Text style={s.brandAccent}>Fusion</Text></Text>
        <Text style={s.tagline}>Agentic Medical AI Platform</Text>
        <View style={s.divider} />
        <View style={s.statusRow}>
          <Animated.View style={[s.statusDot, { transform: [{ scale: pulseAnim }] }]} />
          <Text style={s.statusText}>Secure AI Engine Initializing...</Text>
        </View>
        <View style={s.healthGrid}>
          {[{ label: 'Backend API', ok: true }, { label: 'Ollama LLM', ok: true }, { label: 'Knowledge Graph', ok: true }].map((item) => (
            <View key={item.label} style={s.healthChip}>
              <Text style={[s.healthDot, { color: item.ok ? theme.success : theme.destructive }]}>●</Text>
              <Text style={s.healthLabel}>{item.label}</Text>
            </View>
          ))}
        </View>
      </Animated.View>
      <Animated.View style={[s.ctaContainer, { opacity: fadeAnim }]}>
        <TouchableOpacity style={s.ctaButton} onPress={() => navigation.navigate('Login')} activeOpacity={0.85}>
          <Text style={s.ctaText}>Initialize Session  →</Text>
        </TouchableOpacity>
        <Text style={s.versionText}>v1.0.0 · Clinical Build</Text>
      </Animated.View>
    </View>
  );
}

const styles = (theme: ReturnType<typeof useTheme>['theme']) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.backgroundDeep, alignItems: 'center', justifyContent: 'center', paddingHorizontal: spacing.lg },
  content: { alignItems: 'center', flex: 1, justifyContent: 'center' },
  logoContainer: { marginBottom: spacing.lg },
  logoBox: {
    width: 64,
    height: 64,
    borderRadius: radius.lg,
    backgroundColor: theme.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: theme.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 8,
  },
  brandName: { fontSize: typography['3xl'], fontWeight: '700', color: theme.foreground, letterSpacing: -0.5, fontFamily: fontFamily.extraBold },
  brandAccent: { color: theme.primary },
  tagline: { fontSize: typography.sm, color: theme.mutedForeground, marginTop: spacing.xs, letterSpacing: 1, textTransform: 'uppercase', fontFamily: fontFamily.medium },
  divider: { width: 60, height: 1, backgroundColor: theme.cardBorder, marginVertical: spacing.xl },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.lg },
  statusDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: theme.primary },
  statusText: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.regular },
  healthGrid: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap', justifyContent: 'center' },
  healthChip: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.card, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.full, paddingHorizontal: spacing.md, paddingVertical: spacing.xs, gap: 4 },
  healthDot: { fontSize: 8 },
  healthLabel: { color: theme.foreground, fontSize: typography.xs, fontFamily: fontFamily.medium },
  ctaContainer: { width: '100%', alignItems: 'center', paddingBottom: spacing.xxl },
  ctaButton: { width: '100%', backgroundColor: theme.primary, height: 56, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', shadowColor: theme.primary, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.4, shadowRadius: 12, elevation: 8 },
  ctaText: { color: theme.primaryForeground, fontWeight: '700', fontSize: typography.lg, fontFamily: fontFamily.bold },
  versionText: { color: theme.mutedForeground, fontSize: typography.xs, marginTop: spacing.md, fontFamily: fontFamily.regular },
});
