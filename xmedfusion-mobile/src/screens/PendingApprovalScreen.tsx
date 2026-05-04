import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, ScrollView } from 'react-native';
import { Activity, Clock, LogOut, RefreshCw, ShieldOff } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { glassPanel, mutedCard, shellBackground, surfaceCard } from '../theme/ui';
import { useAuth } from '../theme/AuthContext';

export default function PendingApprovalScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const { isRejected, isApproved, isAdmin, session, signOut, refreshRole, loading, roleLoading } = useAuth();
  const s = styles(theme);

  useEffect(() => {
    if (loading || roleLoading || !session) {
      return;
    }

    if (isApproved) {
      navigation.navigate(isAdmin ? '/admin' : '/dashboard');
    }
  }, [isAdmin, isApproved, loading, navigation, roleLoading, session]);

  useEffect(() => {
    if (loading || roleLoading) {
      return;
    }

    if (!session) {
      navigation.navigate('/login');
    }
  }, [loading, navigation, roleLoading, session]);

  const handleSignOut = async () => {
    await signOut();
    navigation.navigate('/login');
  };

  if (loading || roleLoading) {
    return (
      <View style={s.container}>
        <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
        <View style={s.loadingWrap}>
          <Text style={s.loadingText}>Loading...</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView contentInsetAdjustmentBehavior="automatic" contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.logoWrap}>
          <View style={s.logoBox}>
            <Activity color={theme.primaryForeground} size={24} />
          </View>
          <Text style={s.brandName}>XMed<Text style={s.brandAccent}>Fusion</Text></Text>
        </View>

        <View style={s.card}>
          <View style={s.imagePanel}>
            <Text style={s.imagePanelText}>Secure review</Text>
            <Text style={s.imagePanelSub}>Credential approval required</Text>
          </View>

          {isRejected ? (
            <View style={s.centerBlock}>
              <View style={s.iconCircleDestructive}>
                <ShieldOff color={theme.destructive} size={32} />
              </View>
              <Text style={s.title}>Registration Rejected</Text>
              <Text style={s.copy}>
                Your registration request has been reviewed and was not approved. Please contact the hospital administrator for more information.
              </Text>
            </View>
          ) : (
            <View style={s.centerBlock}>
              <View style={s.iconCircleWarning}>
                <Clock color={theme.warning} size={32} />
              </View>
              <View style={s.statusPill}>
                <View style={s.dot} />
                <Text style={s.statusText}>Credential review in progress</Text>
              </View>
              <Text style={s.title}>Awaiting Approval</Text>
              <Text style={s.copy}>
                Your registration is under review. A platform administrator will verify your credentials and approve your account shortly.
              </Text>
              <View style={s.notice}>
                <Text style={s.noticeText}>
                  You will be able to access the platform once your account is approved. Please check back later.
                </Text>
              </View>
            </View>
          )}

          <View style={s.actions}>
            <TouchableOpacity style={s.primaryButton} onPress={refreshRole}>
              <RefreshCw color={theme.primaryForeground} size={16} />
              <Text style={s.primaryButtonText}>Check Status</Text>
            </TouchableOpacity>
            <TouchableOpacity style={s.secondaryButton} onPress={handleSignOut}>
              <LogOut color={theme.mutedForeground} size={16} />
              <Text style={s.secondaryButtonText}>Sign Out</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) =>
  StyleSheet.create({
    container: { flex: 1, ...shellBackground(theme) },
    scroll: { flexGrow: 1, paddingHorizontal: spacing.lg, paddingVertical: spacing.xl, gap: spacing.lg },
    loadingWrap: { flex: 1, alignItems: 'center', justifyContent: 'center' },
    loadingText: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.regular },
    logoWrap: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, alignSelf: 'center', marginBottom: spacing.xs },
    logoBox: { width: 44, height: 44, borderRadius: radius.md, backgroundColor: theme.primary, alignItems: 'center', justifyContent: 'center' },
    brandName: { fontSize: typography.xl, color: theme.foreground, fontFamily: fontFamily.extraBold },
    brandAccent: { color: theme.primary },
    card: { ...glassPanel(theme), padding: spacing.lg, gap: spacing.md },
    imagePanel: { ...surfaceCard(theme), minHeight: 160, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.backgroundDeep },
    imagePanelText: { color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.bold },
    imagePanelSub: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.regular, marginTop: 4 },
    centerBlock: { alignItems: 'center', gap: spacing.md },
    iconCircleDestructive: { width: 64, height: 64, borderRadius: 20, backgroundColor: theme.destructiveBg, alignItems: 'center', justifyContent: 'center' },
    iconCircleWarning: { width: 64, height: 64, borderRadius: 20, backgroundColor: theme.warningBg, alignItems: 'center', justifyContent: 'center' },
    statusPill: { ...mutedCard(theme), alignSelf: 'center', flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: spacing.sm, paddingVertical: 6 },
    dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: theme.warning },
    statusText: { color: theme.warning, fontSize: typography.xs, fontFamily: fontFamily.bold, textTransform: 'uppercase' },
    title: { color: theme.foreground, fontSize: typography['2xl'], fontFamily: fontFamily.extraBold, textAlign: 'center' },
    copy: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular, textAlign: 'center' },
    notice: { borderWidth: 1, borderColor: theme.warning, backgroundColor: theme.warningBg, borderRadius: radius.lg, padding: spacing.md, width: '100%' },
    noticeText: { color: theme.warning, fontSize: typography.xs, lineHeight: 18, fontFamily: fontFamily.regular, textAlign: 'center' },
    actions: { gap: spacing.sm, marginTop: spacing.xs },
    primaryButton: { backgroundColor: theme.primary, borderRadius: radius.lg, height: 52, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
    primaryButtonText: { color: theme.primaryForeground, fontSize: typography.sm, fontFamily: fontFamily.bold },
    secondaryButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.lg, height: 52, backgroundColor: theme.card },
    secondaryButtonText: { color: theme.mutedForeground, fontSize: typography.sm, fontFamily: fontFamily.semiBold },
  });
