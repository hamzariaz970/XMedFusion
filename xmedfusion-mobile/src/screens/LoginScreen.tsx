import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  StatusBar, KeyboardAvoidingView, Platform, ScrollView,
  ActivityIndicator, Alert
} from 'react-native';
import { Activity, ShieldCheck, User, Settings, Mail, Key, ArrowRight, UserPlus } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { supabase } from '../lib/supabase';

export default function LoginScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [role, setRole] = useState<'radiologist' | 'admin'>('radiologist');
  const s = styles(theme);

  const handleAuth = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    setLoading(true);
    try {
      if (mode === 'signin') {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (error) throw error;
      } else {
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: {
              role: role,
            }
          }
        });
        if (error) throw error;
        Alert.alert('Success', 'Verification email sent! Please check your inbox.');
        setMode('signin');
      }
    } catch (error: any) {
      Alert.alert('Authentication Error', error.message);
    } finally {
      setLoading(false);
    }
  };

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
            <ShieldCheck color={theme.primary} size={14} style={{ marginRight: 6 }} />
            <Text style={s.badgeText}>Clinical Gateway</Text>
          </View>
          <Text style={s.title}>{mode === 'signin' ? 'Initialize Session' : 'Register Account'}</Text>
          <Text style={s.subtitle}>{mode === 'signin' ? 'Authorized medical professionals only' : 'Create your secure medical profile'}</Text>
        </View>

        <View style={s.roleRow}>
          {(['radiologist', 'admin'] as const).map((r) => (
            <TouchableOpacity 
              key={r} 
              style={[s.roleChip, role === r && s.roleChipActive]} 
              onPress={() => setRole(r)}
            >
              <View style={s.roleContent}>
                {r === 'radiologist' ? (
                  <User color={role === r ? theme.primary : theme.mutedForeground} size={18} />
                ) : (
                  <Settings color={role === r ? theme.primary : theme.mutedForeground} size={18} />
                )}
                <Text style={[s.roleChipText, role === r && s.roleChipTextActive]}>
                  {r === 'radiologist' ? 'Radiologist' : 'Admin'}
                </Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>

        <View style={s.form}>
          <View style={s.inputGroup}>
            <Text style={s.inputLabel}>Email Address</Text>
            <View style={s.inputWrapper}>
              <Mail color={theme.mutedForeground} size={18} style={s.inputIcon} />
              <TextInput 
                style={s.input} 
                placeholder="dr@hospital.com" 
                placeholderTextColor={theme.mutedForeground} 
                value={email} 
                onChangeText={setEmail} 
                autoCapitalize="none" 
                keyboardType="email-address" 
              />
            </View>
          </View>
          <View style={s.inputGroup}>
            <Text style={s.inputLabel}>Secure Passkey</Text>
            <View style={s.inputWrapper}>
              <Key color={theme.mutedForeground} size={18} style={s.inputIcon} />
              <TextInput 
                style={s.input} 
                placeholder="••••••••" 
                placeholderTextColor={theme.mutedForeground} 
                value={password} 
                onChangeText={setPassword} 
                secureTextEntry 
              />
            </View>
          </View>
        </View>

        <TouchableOpacity style={s.loginButton} onPress={handleAuth} activeOpacity={0.85} disabled={loading}>
          {loading ? (
            <ActivityIndicator color={theme.primaryForeground} />
          ) : (
            <View style={s.buttonContent}>
              <Text style={s.loginButtonText}>
                {mode === 'signin' ? 'Authenticate' : 'Create Account'}
              </Text>
              {mode === 'signin' ? (
                <ArrowRight color={theme.primaryForeground} size={20} style={{ marginLeft: 8 }} />
              ) : (
                <UserPlus color={theme.primaryForeground} size={20} style={{ marginLeft: 8 }} />
              )}
            </View>
          )}
        </TouchableOpacity>

        <TouchableOpacity style={s.toggleButton} onPress={() => setMode(mode === 'signin' ? 'signup' : 'signin')}>
          <Text style={s.toggleText}>
            {mode === 'signin' ? "Don't have an account? Sign Up" : "Already have an account? Sign In"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={s.biometricButton}>
          <Activity color={theme.mutedForeground} size={18} style={{ marginRight: 10 }} />
          <Text style={s.biometricText}>Use Biometric Authentication</Text>
        </TouchableOpacity>

        <Text style={s.disclaimer}>
          By authenticating, you confirm you are an authorized medical professional. Sessions are logged and encrypted.
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = (theme: any) => StyleSheet.create({
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
  badge: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.primaryGlow, borderWidth: 1, borderColor: theme.primary, borderRadius: radius.full, paddingHorizontal: spacing.md, paddingVertical: spacing.xs, alignSelf: 'flex-start', marginBottom: spacing.md },
  badgeText: { color: theme.primary, fontSize: typography.xs, fontWeight: '600', fontFamily: fontFamily.semiBold },
  title: { fontSize: typography['3xl'], fontWeight: '700', color: theme.foreground, letterSpacing: -0.5, fontFamily: fontFamily.extraBold },
  subtitle: { fontSize: typography.base, color: theme.mutedForeground, marginTop: spacing.xs, fontFamily: fontFamily.regular },
  roleRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.xl },
  roleChip: { flex: 1, paddingVertical: spacing.sm, borderRadius: radius.md, borderWidth: 1, borderColor: theme.cardBorder, backgroundColor: theme.card, alignItems: 'center' },
  roleContent: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  roleChipActive: { borderColor: theme.primary, backgroundColor: theme.primaryGlow },
  roleChipText: { color: theme.mutedForeground, fontSize: typography.sm, fontWeight: '600', fontFamily: fontFamily.semiBold },
  roleChipTextActive: { color: theme.primary },
  form: { gap: spacing.md, marginBottom: spacing.lg },
  inputGroup: { gap: spacing.xs },
  inputLabel: { color: theme.mutedForeground, fontSize: typography.sm, fontWeight: '500', marginLeft: 4, fontFamily: fontFamily.medium },
  inputWrapper: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.card, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.md, paddingHorizontal: spacing.md },
  inputIcon: { marginRight: 10 },
  input: { flex: 1, height: 54, color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.regular },
  loginButton: { height: 56, backgroundColor: theme.primary, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', shadowColor: theme.primary, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.4, shadowRadius: 12, elevation: 8, marginBottom: spacing.md },
  buttonContent: { flexDirection: 'row', alignItems: 'center' },
  loginButtonText: { color: theme.primaryForeground, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold },
  toggleButton: { alignItems: 'center', paddingVertical: spacing.md, marginBottom: spacing.sm },
  toggleText: { color: theme.primary, fontSize: typography.sm, fontWeight: '600', fontFamily: fontFamily.semiBold },
  biometricButton: { flexDirection: 'row', height: 48, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.card, marginBottom: spacing.xl },
  biometricText: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.medium },
  disclaimer: { color: theme.mutedForeground, fontSize: typography.xs, textAlign: 'center', lineHeight: 18, paddingHorizontal: spacing.md, fontFamily: fontFamily.regular },
});
