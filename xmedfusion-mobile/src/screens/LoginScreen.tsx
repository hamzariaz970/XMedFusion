import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { ArrowRight, Activity, ShieldCheck, Mail, Key, UserPlus, Sparkles, Stethoscope, User } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../theme/AuthContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { createShadow, glassPanel, mutedCard, shellBackground, surfaceCard } from '../theme/ui';
import { supabase } from '../lib/supabase';

const SPECIALIZATIONS = [
  'Radiology',
  'Cardiology',
  'Pulmonology',
  'Oncology',
  'General Medicine',
  'Neurology',
  'Orthopedics',
  'Pathology',
];

export default function LoginScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const { session, refreshRole, loading: authLoading, roleLoading, isAdmin, isApproved, isPending, isRejected } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [specialization, setSpecialization] = useState('Radiology');
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [loading, setLoading] = useState(false);
  const s = styles(theme);

  useEffect(() => {
    if (authLoading || roleLoading || !session) {
      return;
    }

    if (isApproved) {
      navigation.navigate(isAdmin ? '/admin' : '/dashboard');
      return;
    }

    if (isPending || isRejected) {
      navigation.navigate('/pending');
    }
  }, [authLoading, isAdmin, isApproved, isPending, isRejected, navigation, roleLoading, session]);

  const deriveFallbackFullName = (emailValue: string | undefined) => {
    const localPart = emailValue?.split('@')[0]?.trim();
    return localPart || 'Doctor';
  };

  const ensureDoctorProfile = async (
    userId: string,
    userEmail: string | undefined,
    preferred?: { fullName?: string; specialization?: string }
  ) => {
    if (!userEmail) return;

    const { data: existingProfile } = await supabase
      .from('doctors')
      .select('id, user_id, full_name, specialization, status')
      .eq('user_id', userId)
      .maybeSingle();

    if (existingProfile) {
      return;
    }

    const { data: emailMatch } = await supabase
      .from('doctors')
      .select('id, user_id, full_name, specialization, status')
      .eq('email', userEmail)
      .maybeSingle();

    const nextFullName = preferred?.fullName?.trim() || emailMatch?.full_name || deriveFallbackFullName(userEmail);
    const nextSpecialization = preferred?.specialization || emailMatch?.specialization || 'Radiology';

    if (emailMatch) {
      const { error: updateErr } = await supabase
        .from('doctors')
        .update({
          user_id: userId,
          full_name: nextFullName,
          specialization: nextSpecialization,
          status: emailMatch.status === 'pre-approved' ? 'active' : (emailMatch.status || 'active'),
        })
        .eq('id', emailMatch.id);

      if (updateErr) {
        console.error('Failed to re-link doctor profile:', updateErr);
      }
      return;
    }

    const { error: insertErr } = await supabase.from('doctors').insert({
      user_id: userId,
      full_name: nextFullName,
      email: userEmail,
      specialization: nextSpecialization,
      status: 'active',
    });

    if (insertErr) {
      console.error('Failed to self-heal doctor profile:', insertErr);
    }
  };

  const handleAuth = async () => {
    if (!email || !password) {
      Alert.alert('Missing details', 'Please fill in your email and password.');
      return;
    }

    setLoading(true);
    try {
      if (mode === 'signup') {
        if (!fullName.trim()) {
          Alert.alert('Missing details', 'Please enter your full name to register.');
          return;
        }

        const { data, error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;

        const userId = data.user?.id;
        if (!userId) {
          Alert.alert('Check your email', 'Verification email sent. Please confirm your account before signing in.');
          return;
        }

        const { data: preApproved } = await supabase
          .from('doctors')
          .select('*')
          .eq('email', email.trim())
          .eq('status', 'pre-approved')
          .maybeSingle();

        const isPreApproved = !!preApproved;

        const { error: roleError } = await supabase
          .from('user_roles')
          .insert({
            user_id: userId,
            role: 'doctor',
            approval_status: isPreApproved ? 'approved' : 'pending',
          });

        if (roleError) {
          console.error('Failed to insert user_role:', roleError);
        }

        if (isPreApproved) {
          const { error: updateErr } = await supabase
            .from('doctors')
            .update({
              user_id: userId,
              status: 'active',
              full_name: fullName.trim(),
              specialization,
            })
            .eq('id', preApproved.id);
          if (updateErr) {
            console.error('Failed to update pre-approved doctor:', updateErr);
          }
        } else {
          const { error: docError } = await supabase.from('doctors').insert({
            user_id: userId,
            full_name: fullName.trim(),
            email: email.trim(),
            specialization,
            status: 'active',
          });
          if (docError) {
            console.error('Failed to insert doctor:', docError);
          }
        }

        await ensureDoctorProfile(userId, email.trim(), {
          fullName,
          specialization,
        });

        await refreshRole();

        if (data.session) {
          if (isPreApproved) {
            Alert.alert('Account created', "You've been pre-approved by an admin.");
            navigation.navigate('/dashboard');
          } else {
            Alert.alert('Pending approval', 'Your registration is pending admin approval.');
            navigation.navigate('/pending');
          }
        } else {
          Alert.alert('Check your email', 'Confirm your email first. Your account still needs admin approval after verification.');
        }
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;

        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          const { data: roleData } = await supabase
            .from('user_roles')
            .select('*')
            .eq('user_id', user.id)
            .maybeSingle();

          if (roleData?.role === 'doctor') {
            await ensureDoctorProfile(user.id, user.email);
          }

          await refreshRole();

          if (!roleData || roleData.approval_status === 'pending') {
            Alert.alert('Pending approval', 'Your account is pending admin approval.');
            navigation.navigate('/pending');
          } else if (roleData.approval_status === 'rejected') {
            Alert.alert('Access denied', 'Your registration has been rejected.');
            navigation.navigate('/pending');
          } else if (roleData.role === 'admin') {
            Alert.alert('Welcome back', 'Welcome back, Admin!');
            navigation.navigate('/admin');
          } else {
            Alert.alert('Welcome back', 'Welcome back to XMedFusion!');
            navigation.navigate('/dashboard');
          }
        }
      }
    } catch (error: any) {
      Alert.alert('Authentication error', error.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView style={s.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        contentContainerStyle={s.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={s.heroCard}>
          <View style={s.logoRow}>
            <View style={s.logoBox}>
              <Activity color={theme.primaryForeground} size={24} />
            </View>
            <Text style={s.brandName}>
              XMed<Text style={s.brandAccent}>Fusion</Text>
            </Text>
          </View>
          <View style={s.badge}>
            <ShieldCheck color={theme.primary} size={14} />
            <Text style={s.badgeText}>Secure access</Text>
          </View>
          <Text style={s.title}>{mode === 'signin' ? 'Sign in to your account' : 'Register as a Doctor'}</Text>
          <Text style={s.subtitle}>
            {mode === 'signin'
              ? 'Clinical AI, gated for verified care teams.'
              : 'Your account will be reviewed by an administrator before access is granted.'}
          </Text>
          <View style={s.heroPills}>
            <View style={s.heroPill}>
              <Sparkles color={theme.primary} size={14} />
              <Text style={s.heroPillText}>Role-aware routing</Text>
            </View>
            <View style={s.heroPill}>
              <Stethoscope color={theme.primary} size={14} />
              <Text style={s.heroPillText}>Doctor specialization capture</Text>
            </View>
          </View>
        </View>

        <View style={s.formCard}>
          <View style={s.cardHeader}>
            <Text style={s.cardTitle}>{mode === 'signin' ? 'Secure Portal' : 'Doctor Registration'}</Text>
            <Text style={s.cardDescription}>
              {mode === 'signin'
                ? 'Enter your credentials to access the medical diagnostic agent.'
                : 'Fill in your details to request platform access.'}
            </Text>
          </View>

          {mode === 'signup' && (
            <>
              <View style={s.inputGroup}>
                <Text style={s.inputLabel}>Full Name</Text>
                <View style={s.inputWrapper}>
                  <User color={theme.mutedForeground} size={18} style={s.inputIcon} />
                  <TextInput
                    style={s.input}
                    placeholder="Dr. Jane Doe"
                    placeholderTextColor={theme.mutedForeground}
                    value={fullName}
                    onChangeText={setFullName}
                  />
                </View>
              </View>

              <View style={s.inputGroup}>
                <Text style={s.inputLabel}>Specialization</Text>
                <View style={s.inputWrapper}>
                  <Stethoscope color={theme.mutedForeground} size={18} style={s.inputIcon} />
                  <TextInput
                    style={s.input}
                    placeholder="Radiology"
                    placeholderTextColor={theme.mutedForeground}
                    value={specialization}
                    onChangeText={setSpecialization}
                  />
                </View>
                <View style={s.specializationRow}>
                  {SPECIALIZATIONS.slice(0, 4).map((spec) => (
                    <TouchableOpacity
                      key={spec}
                      style={[s.specChip, specialization === spec && s.specChipActive]}
                      onPress={() => setSpecialization(spec)}
                    >
                      <Text style={[s.specChipText, specialization === spec && s.specChipTextActive]}>{spec}</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            </>
          )}

          <View style={s.form}>
            <View style={s.inputGroup}>
              <Text style={s.inputLabel}>Email Address</Text>
              <View style={s.inputWrapper}>
                <Mail color={theme.mutedForeground} size={18} style={s.inputIcon} />
                <TextInput
                  style={s.input}
                  placeholder="name@hospital.com"
                  placeholderTextColor={theme.mutedForeground}
                  value={email}
                  onChangeText={setEmail}
                  autoCapitalize="none"
                  keyboardType="email-address"
                />
              </View>
            </View>

            <View style={s.inputGroup}>
              <Text style={s.inputLabel}>Password</Text>
              <View style={s.inputWrapper}>
                <Key color={theme.mutedForeground} size={18} style={s.inputIcon} />
                <TextInput
                  style={s.input}
                  placeholder="Enter your password"
                  placeholderTextColor={theme.mutedForeground}
                  value={password}
                  onChangeText={setPassword}
                  secureTextEntry
                />
              </View>
            </View>
          </View>

          <TouchableOpacity style={s.loginButton} onPress={handleAuth} activeOpacity={0.9} disabled={loading}>
            {loading ? (
              <ActivityIndicator color={theme.primaryForeground} />
            ) : (
              <View style={s.buttonContent}>
                <Text style={s.loginButtonText}>{mode === 'signin' ? 'Sign In' : 'Submit Registration'}</Text>
                <ArrowRight color={theme.primaryForeground} size={20} style={{ marginLeft: 8 }} />
              </View>
            )}
          </TouchableOpacity>

          <TouchableOpacity style={s.toggleButton} onPress={() => setMode(mode === 'signin' ? 'signup' : 'signin')}>
            <Text style={s.toggleText}>
              {mode === 'signin' ? "Don't have an account? Register" : 'Already have an account? Sign in'}
            </Text>
          </TouchableOpacity>
        </View>

        <Text style={s.disclaimer}>
          By continuing, you confirm you are an authorized medical professional using a secured clinical system.
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = (theme: any) =>
  StyleSheet.create({
    container: { flex: 1, ...shellBackground(theme) },
    scrollContent: { flexGrow: 1, paddingHorizontal: spacing.lg, paddingVertical: spacing.xl, gap: spacing.lg },
    heroCard: { ...glassPanel(theme), padding: spacing.lg, gap: spacing.md },
    logoRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
    logoBox: {
      width: 44,
      height: 44,
      borderRadius: radius.md,
      backgroundColor: theme.primary,
      alignItems: 'center',
      justifyContent: 'center',
      ...createShadow(theme, 'md'),
    },
    brandName: { fontSize: typography.xl, color: theme.foreground, fontFamily: fontFamily.extraBold },
    brandAccent: { color: theme.primary },
    badge: {
      ...mutedCard(theme),
      alignSelf: 'flex-start',
      flexDirection: 'row',
      alignItems: 'center',
      gap: 6,
      paddingHorizontal: spacing.sm,
      paddingVertical: 6,
    },
    badgeText: { color: theme.primary, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
    title: { color: theme.foreground, fontSize: typography['2xl'], lineHeight: 34, fontFamily: fontFamily.extraBold },
    subtitle: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
    heroPills: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap' },
    heroPill: {
      ...mutedCard(theme),
      flexDirection: 'row',
      alignItems: 'center',
      gap: 6,
      paddingHorizontal: spacing.sm,
      paddingVertical: 8,
    },
    heroPillText: { color: theme.foreground, fontSize: typography.xs, fontFamily: fontFamily.semiBold },
    formCard: { ...surfaceCard(theme), padding: spacing.lg, gap: spacing.md },
    cardHeader: { gap: 4 },
    cardTitle: { color: theme.foreground, fontSize: typography.lg, fontFamily: fontFamily.bold },
    cardDescription: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 20, fontFamily: fontFamily.regular },
    form: { gap: spacing.md },
    inputGroup: { gap: spacing.xs },
    inputLabel: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.medium, marginLeft: 4 },
    inputWrapper: { ...mutedCard(theme), flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.md },
    inputIcon: { marginRight: 10 },
    input: { flex: 1, height: 54, color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.regular },
    specializationRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 6 },
    specChip: {
      borderWidth: 1,
      borderColor: theme.cardBorder,
      borderRadius: radius.full,
      paddingHorizontal: 10,
      paddingVertical: 6,
      backgroundColor: theme.card,
    },
    specChipActive: { backgroundColor: theme.primary, borderColor: theme.primary },
    specChipText: { color: theme.foreground, fontSize: typography.xs, fontFamily: fontFamily.semiBold },
    specChipTextActive: { color: theme.primaryForeground },
    loginButton: { height: 56, backgroundColor: theme.primary, borderRadius: radius.lg, alignItems: 'center', justifyContent: 'center', ...createShadow(theme, 'lg') },
    buttonContent: { flexDirection: 'row', alignItems: 'center' },
    loginButtonText: { color: theme.primaryForeground, fontSize: typography.base, fontFamily: fontFamily.bold },
    toggleButton: { alignItems: 'center', paddingTop: spacing.xs },
    toggleText: { color: theme.primary, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
    disclaimer: { color: theme.mutedForeground, fontSize: typography.xs, textAlign: 'center', lineHeight: 18, paddingHorizontal: spacing.md, fontFamily: fontFamily.regular },
  });
