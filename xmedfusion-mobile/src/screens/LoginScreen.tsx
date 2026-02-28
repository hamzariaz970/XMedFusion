import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  StatusBar, KeyboardAvoidingView, Platform, ScrollView,
} from 'react-native';
import { Activity } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

export default function LoginScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<'radiologist' | 'admin'>('radiologist');
  const s = styles(theme);

  return (
    <KeyboardAvoidingView style={s.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView contentContainerStyle={s.scrollContent} keyboardShouldPersistTaps="handled">
        <View style={s.header}>
          <View style={s.logoRow}>
            <View style={s.logoBox}>
               <Activity color={theme.primaryForeground} size={24} />
            </View>
            <Text style={s.brandName}>XMed<Text style={s.brandAccent}>Fusion</Text></Text>
          </View>
          <View style={s.badge}>
            <Text style={s.badgeText}>🔒 Clinical Gateway</Text>
          </View>
          <Text style={s.title}>Initialize Session</Text>
          <Text style={s.subtitle}>Authorized medical professionals only</Text>
        </View>

        <View style={s.roleRow}>
          {(['radiologist', 'admin'] as const).map((r) => (
            <TouchableOpacity key={r} style={[s.roleChip, role === r && s.roleChipActive]} onPress={() => setRole(r)}>
              <Text style={[s.roleChipText, role === r && s.roleChipTextActive]}>
                {r === 'radiologist' ? '🩻 Radiologist' : '⚙️ Admin'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={s.form}>
          <View style={s.inputGroup}>
            <Text style={s.inputLabel}>Hospital ID or Email</Text>
            <TextInput style={s.input} placeholder="e.g. DR-001 or dr@hospital.com" placeholderTextColor={theme.mutedForeground} value={email} onChangeText={setEmail} autoCapitalize="none" keyboardType="email-address" />
          </View>
          <View style={s.inputGroup}>
            <Text style={s.inputLabel}>Secure Passkey</Text>
            <TextInput style={s.input} placeholder="••••••••" placeholderTextColor={theme.mutedForeground} value={password} onChangeText={setPassword} secureTextEntry />
          </View>
        </View>

        <TouchableOpacity style={s.loginButton} onPress={() => navigation.replace('MainTabs')} activeOpacity={0.85}>
          <Text style={s.loginButtonText}>Authenticate  →</Text>
        </TouchableOpacity>

        <TouchableOpacity style={s.biometricButton}>
          <Text style={s.biometricText}>👆  Use Biometric Authentication</Text>
        </TouchableOpacity>

        <Text style={s.disclaimer}>
          By authenticating, you confirm you are an authorized medical professional. Sessions are logged and encrypted.
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = (theme: ReturnType<typeof useTheme>['theme']) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.backgroundDeep },
  scrollContent: { flexGrow: 1, paddingHorizontal: spacing.lg, paddingTop: spacing.xxl + spacing.md, paddingBottom: spacing.xxl },
  header: { marginBottom: spacing.xl },
  logoRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.lg },
  logoBox: {
    width: 42,
    height: 42,
    borderRadius: radius.md,
    backgroundColor: theme.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: theme.primary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 4,
  },
  brandName: { fontSize: typography.xl, fontWeight: '700', color: theme.foreground, letterSpacing: -0.5, fontFamily: fontFamily.extraBold },
  brandAccent: { color: theme.primary },
  badge: { backgroundColor: theme.primaryGlow, borderWidth: 1, borderColor: theme.primary, borderRadius: radius.full, paddingHorizontal: spacing.md, paddingVertical: spacing.xs, alignSelf: 'flex-start', marginBottom: spacing.md },
  badgeText: { color: theme.primary, fontSize: typography.xs, fontWeight: '600', fontFamily: fontFamily.semiBold },
  title: { fontSize: typography['3xl'], fontWeight: '700', color: theme.foreground, letterSpacing: -0.5, fontFamily: fontFamily.extraBold },
  subtitle: { fontSize: typography.base, color: theme.mutedForeground, marginTop: spacing.xs, fontFamily: fontFamily.regular },
  roleRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.xl },
  roleChip: { flex: 1, paddingVertical: spacing.sm + 2, borderRadius: radius.md, borderWidth: 1, borderColor: theme.cardBorder, backgroundColor: theme.card, alignItems: 'center' },
  roleChipActive: { borderColor: theme.primary, backgroundColor: theme.primaryGlow },
  roleChipText: { color: theme.mutedForeground, fontSize: typography.sm, fontWeight: '600', fontFamily: fontFamily.semiBold },
  roleChipTextActive: { color: theme.primary },
  form: { gap: spacing.md, marginBottom: spacing.lg },
  inputGroup: { gap: spacing.xs },
  inputLabel: { color: theme.mutedForeground, fontSize: typography.sm, fontWeight: '500', marginLeft: 4, fontFamily: fontFamily.medium },
  input: { height: 54, backgroundColor: theme.card, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.md, paddingHorizontal: spacing.md, color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.regular },
  loginButton: { height: 56, backgroundColor: theme.primary, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', shadowColor: theme.primary, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.4, shadowRadius: 12, elevation: 8, marginBottom: spacing.md },
  loginButtonText: { color: theme.primaryForeground, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold },
  biometricButton: { height: 48, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.card, marginBottom: spacing.xl },
  biometricText: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.medium },
  disclaimer: { color: theme.mutedForeground, fontSize: typography.xs, textAlign: 'center', lineHeight: 18, paddingHorizontal: spacing.md, fontFamily: fontFamily.regular },
});
