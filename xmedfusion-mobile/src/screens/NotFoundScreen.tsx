import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar } from 'react-native';
import { Activity, ArrowLeft, FileSearch } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { glassPanel, shellBackground } from '../theme/ui';

export default function NotFoundScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const s = styles(theme);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <View style={s.wrapper}>
        <View style={s.card}>
          <View style={s.iconBox}>
            <FileSearch color={theme.primary} size={34} />
          </View>
          <Text style={s.eyebrow}>404 diagnostic miss</Text>
          <Text style={s.title}>Page not found</Text>
          <Text style={s.copy}>The requested route is not part of the current XMedFusion clinical workspace.</Text>
          <View style={s.actions}>
            <TouchableOpacity style={s.primaryButton} onPress={() => navigation.navigate('/')}>
              <ArrowLeft color={theme.primaryForeground} size={16} />
              <Text style={s.primaryButtonText}>Return Home</Text>
            </TouchableOpacity>
            <TouchableOpacity style={s.secondaryButton} onPress={() => navigation.navigate('/knowledge-graph')}>
              <Activity color={theme.foreground} size={16} />
              <Text style={s.secondaryButtonText}>Open Graph</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </View>
  );
}

const styles = (theme: any) =>
  StyleSheet.create({
    container: { flex: 1, ...shellBackground(theme), justifyContent: 'center' },
    wrapper: { paddingHorizontal: spacing.lg },
    card: { ...glassPanel(theme), padding: spacing.lg, alignItems: 'center', gap: spacing.sm },
    iconBox: { width: 68, height: 68, borderRadius: radius.lg, backgroundColor: theme.primaryGlow, alignItems: 'center', justifyContent: 'center' },
    eyebrow: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase', letterSpacing: 1 },
    title: { color: theme.foreground, fontSize: typography['2xl'], fontFamily: fontFamily.extraBold },
    copy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, textAlign: 'center', fontFamily: fontFamily.regular },
    actions: { width: '100%', gap: spacing.sm, marginTop: spacing.sm },
    primaryButton: { backgroundColor: theme.primary, borderRadius: radius.lg, height: 52, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
    primaryButtonText: { color: theme.primaryForeground, fontSize: typography.sm, fontFamily: fontFamily.bold },
    secondaryButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.lg, height: 52, backgroundColor: theme.card },
    secondaryButtonText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
  });
