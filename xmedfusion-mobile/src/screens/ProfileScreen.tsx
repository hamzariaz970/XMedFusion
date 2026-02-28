import React from 'react';
import { View, Text, StyleSheet, ScrollView, StatusBar, TouchableOpacity, Image } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { 
  ChevronLeft, 
  Camera, 
  Edit3, 
  User, 
  Mail, 
  Briefcase, 
  Hospital, 
  BadgeCheck, 
  MapPin,
  Calendar,
  LogOut
} from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { useAuth } from '../theme/AuthContext';

export default function ProfileScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark } = useTheme();
  const { user, signOut } = useAuth();
  const s = styles(theme);

  const profile = {
    name: user?.email?.split('@')[0].replace('.', ' ').replace(/(^\w|\s\w)/g, m => m.toUpperCase()) || 'Dr. Ahmad Riaz',
    role: user?.user_metadata?.role === 'admin' ? 'System Administrator' : 'Senior Radiologist',
    hospital: 'PKH General Hospital',
    email: user?.email || 'ahmad.riaz@xmedfusion.ai',
    id: user?.id?.substring(0, 8).toUpperCase() || 'DR-99212',
    memberSince: user?.created_at ? new Date(user.created_at).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }) : 'Oct 2024',
    location: 'Lahore, Pakistan',
    specialization: user?.user_metadata?.role === 'admin' ? 'IT Operations' : 'Agentic AI Workflows'
  };

  const handleSignOut = async () => {
    try {
      await signOut();
      // Navigation handled by AppNavigator
    } catch (error: any) {
      alert('Error signing out');
    }
  };

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      
      {/* Header */}
      <View style={s.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <ChevronLeft color={theme.foreground} size={24} />
        </TouchableOpacity>
        <Text style={s.headerTitle}>Public Profile</Text>
        <TouchableOpacity onPress={() => alert('Profile Editing coming soon.')}>
           <Edit3 color={theme.primary} size={20} />
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        {/* Profile Header */}
        <View style={s.profileHeader}>
           <View style={s.avatarWrapper}>
              <View style={s.avatarRing}>
                 <User color={theme.primary} size={50} strokeWidth={1} />
              </View>
              <TouchableOpacity style={s.cameraBtn} onPress={() => alert('Accessing Camera/Gallery...')}>
                 <Camera color={theme.white} size={14} />
              </TouchableOpacity>
           </View>
           <Text style={s.nameText}>{profile.name}</Text>
           <View style={s.roleBadge}>
              <BadgeCheck color={theme.primary} size={14} />
              <Text style={s.roleText}>{profile.role}</Text>
           </View>
        </View>

        {/* Details Grid */}
        <View style={s.infoSection}>
           <Text style={s.sectionTitle}>Account Information</Text>
           <View style={s.infoCard}>
              {[
                { icon: Hospital, label: 'Institution', value: profile.hospital },
                { icon: Mail, label: 'Work Email', value: profile.email },
                { icon: Briefcase, label: 'Specialization', value: profile.specialization },
                { icon: MapPin, label: 'Location', value: profile.location },
                { icon: Calendar, label: 'Member Since', value: profile.memberSince },
              ].map((item, ix) => (
                <View key={ix} style={[s.infoRow, ix > 0 && s.infoRowBorder]}>
                   <View style={s.iconCircle}>
                      <item.icon color={theme.mutedForeground} size={18} />
                   </View>
                   <View>
                      <Text style={s.itemLabel}>{item.label}</Text>
                      <Text style={s.itemValue}>{item.value}</Text>
                   </View>
                </View>
              ))}
           </View>
        </View>

        <TouchableOpacity style={s.signOutBtn} onPress={handleSignOut}>
           <LogOut color={theme.destructive} size={20} />
           <Text style={s.signOutText}>Sign Out of Session</Text>
        </TouchableOpacity>

        <TouchableOpacity style={s.verifyDocsBtn} onPress={() => alert('Opening Secure Document Viewer...')}>
           <Text style={s.verifyText}>View Credentials & Verification Documents</Text>
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
  
  profileHeader: { alignItems: 'center', gap: 10, marginTop: spacing.md },
  avatarWrapper: { position: 'relative' },
  avatarRing: { width: 100, height: 100, borderRadius: 50, backgroundColor: theme.primaryGlow, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: theme.primary },
  cameraBtn: { position: 'absolute', bottom: 2, right: 2, backgroundColor: theme.primary, width: 30, height: 30, borderRadius: 15, alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: theme.card },
  nameText: { color: theme.foreground, fontSize: typography['2xl'], fontWeight: '700', fontFamily: fontFamily.bold },
  roleBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: theme.primaryGlow, paddingHorizontal: 12, paddingVertical: 6, borderRadius: radius.full },
  roleText: { color: theme.primary, fontWeight: '700', fontSize: typography.xs, fontFamily: fontFamily.bold },

  infoSection: { gap: spacing.md },
  sectionTitle: { color: theme.foreground, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold, paddingLeft: 4 },
  infoCard: { backgroundColor: theme.card, borderRadius: radius.lg, borderWidth: 1, borderColor: theme.cardBorder, overflow: 'hidden' },
  infoRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, padding: spacing.md },
  infoRowBorder: { borderTopWidth: 1, borderTopColor: theme.cardBorder },
  iconCircle: { width: 36, height: 36, borderRadius: 18, backgroundColor: theme.backgroundDeep, alignItems: 'center', justifyContent: 'center' },
  itemLabel: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular, marginBottom: 2 },
  itemValue: { color: theme.foreground, fontSize: typography.sm, fontWeight: '600', fontFamily: fontFamily.semiBold },

  signOutBtn: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    justifyContent: 'center', 
    gap: 10, 
    padding: spacing.md, 
    backgroundColor: theme.destructiveBg, 
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: theme.destructive + '40'
  },
  signOutText: { color: theme.destructive, fontSize: typography.sm, fontWeight: '700', fontFamily: fontFamily.bold },
  
  verifyDocsBtn: { alignItems: 'center', padding: spacing.md },
  verifyText: { color: theme.primary, fontSize: typography.sm, fontWeight: '600', fontFamily: fontFamily.semiBold, textDecorationLine: 'underline' },
});
