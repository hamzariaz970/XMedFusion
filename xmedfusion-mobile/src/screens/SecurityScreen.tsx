import React from 'react';
import { View, Text, StyleSheet, ScrollView, StatusBar, TouchableOpacity, Switch } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { 
  ChevronLeft, 
  Shield, 
  Lock, 
  Fingerprint, 
  Smartphone, 
  Eye, 
  Trash2,
  ChevronRight,
  ShieldCheck
} from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

export default function SecurityScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const s = styles(theme);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      
      {/* Header */}
      <View style={s.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <ChevronLeft color={theme.foreground} size={24} />
        </TouchableOpacity>
        <Text style={s.headerTitle}>Security & Access</Text>
        <View style={{ width: 24 }} />
      </View>

      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.securityBanner}>
           <Shield color={theme.success} size={48} strokeWidth={1.5} />
           <Text style={s.bannerTitle}>Account Secured</Text>
           <Text style={s.bannerSubtitle}>Your clinical data is protected with 256-bit AES encryption.</Text>
        </View>

        <View style={s.section}>
           <Text style={s.sectionTitle}>Access Control</Text>
           <View style={s.card}>
              <TouchableOpacity style={s.row}>
                 <View style={s.iconBg}>
                    <Lock color={theme.primary} size={20} />
                 </View>
                 <View style={s.rowContent}>
                    <Text style={s.rowLabel}>Change Password</Text>
                    <Text style={s.rowSub}>Last changed 3 months ago</Text>
                 </View>
                 <ChevronRight color={theme.mutedForeground} size={18} />
              </TouchableOpacity>

              <View style={[s.row, s.rowBorder]}>
                 <View style={s.iconBg}>
                    <Fingerprint color={theme.primary} size={20} />
                 </View>
                 <View style={s.rowContent}>
                    <Text style={s.rowLabel}>Biometric Login</Text>
                    <Text style={s.rowSub}>FaceID or Fingerprint</Text>
                 </View>
                 <Switch value={true} trackColor={{ true: theme.primary }} />
              </View>

              <View style={[s.row, s.rowBorder]}>
                 <View style={s.iconBg}>
                    <Smartphone color={theme.primary} size={20} />
                 </View>
                 <View style={s.rowContent}>
                    <Text style={s.rowLabel}>Two-Factor Auth</Text>
                    <Text style={s.rowSub}>SMS/Authenticator app</Text>
                 </View>
                 <Switch value={false} trackColor={{ true: theme.primary }} />
              </View>
           </View>
        </View>

        <View style={s.section}>
           <Text style={s.sectionTitle}>Privacy</Text>
           <View style={s.card}>
              <TouchableOpacity style={s.row}>
                 <View style={s.iconBg}>
                    <Eye color={theme.primary} size={20} />
                 </View>
                 <View style={s.rowContent}>
                    <Text style={s.rowLabel}>Login Activity</Text>
                    <Text style={s.rowSub}>View recent session locations</Text>
                 </View>
                 <ChevronRight color={theme.mutedForeground} size={18} />
              </TouchableOpacity>
           </View>
        </View>

        <TouchableOpacity style={s.deleteBtn}>
           <Trash2 color={theme.destructive} size={18} />
           <Text style={s.deleteText}>Request Data Deletion</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.background },
  header: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    justifyContent: 'space-between', 
    paddingHorizontal: spacing.lg, 
    paddingTop: spacing.xxl, 
    paddingBottom: spacing.md,
    backgroundColor: theme.card,
    borderBottomWidth: 1,
    borderBottomColor: theme.cardBorder
  },
  headerTitle: { color: theme.foreground, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold },
  scroll: { padding: spacing.lg, gap: spacing.xl, paddingBottom: spacing.xxl },

  securityBanner: { alignItems: 'center', gap: 10, paddingVertical: spacing.xl },
  bannerTitle: { color: theme.foreground, fontSize: typography.xl, fontWeight: '700', fontFamily: fontFamily.bold },
  bannerSubtitle: { color: theme.mutedForeground, fontSize: typography.sm, textAlign: 'center', fontFamily: fontFamily.regular, maxWidth: '80%' },

  section: { gap: spacing.md },
  sectionTitle: { color: theme.foreground, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold, paddingLeft: 4 },
  card: { backgroundColor: theme.card, borderRadius: radius.lg, borderWidth: 1, borderColor: theme.cardBorder, overflow: 'hidden' },
  row: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, padding: spacing.md },
  rowBorder: { borderTopWidth: 1, borderTopColor: theme.cardBorder },
  iconBg: { width: 40, height: 40, borderRadius: 20, backgroundColor: theme.backgroundDeep, alignItems: 'center', justifyContent: 'center' },
  rowContent: { flex: 1 },
  rowLabel: { color: theme.foreground, fontWeight: '600', fontSize: typography.base, fontFamily: fontFamily.semiBold },
  rowSub: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular, marginTop: 2 },

  deleteBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, marginTop: spacing.lg },
  deleteText: { color: theme.destructive, fontSize: typography.sm, fontWeight: '600', fontFamily: fontFamily.semiBold },
});
