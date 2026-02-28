import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, ScrollView } from 'react-native';
import { 
  User, 
  Lock, 
  Hospital, 
  Brain, 
  Settings as SettingsIcon, 
  Thermometer, 
  Radio, 
  Cpu, 
  History,
  LogOut,
  Sun,
  Moon,
  ChevronRight
} from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { useNavigation } from '@react-navigation/native';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

const settingsSections = [
  {
    title: 'Account',
    items: [
      { icon: User, label: 'Profile & Role', sublabel: 'Dr. Ahmad · Radiologist', target: 'Profile' },
      { icon: Lock, label: 'Security & 2FA', sublabel: 'Last login: Today 3:14 PM', target: 'Security' },
      { icon: Hospital, label: 'Hospital Configuration', sublabel: 'PKH General Hospital', target: 'Hospital' },
    ],
  },
  {
    title: 'AI Configuration',
    items: [
      { icon: Brain, label: 'Active Model', sublabel: 'XMedFusion Agentic Pipeline', target: 'ModelSettings' },
      { icon: SettingsIcon, label: 'Report Verbosity', sublabel: 'Detailed + Recommendations', target: 'VerbositySettings' },
      { icon: Thermometer, label: 'LLM Temperature', sublabel: '0.3 (Clinical Mode)', target: 'TempSettings' },
    ],
  },
  {
    title: 'System',
    items: [
      { icon: Radio, label: 'Backend API', sublabel: 'Connected (Healthy)', target: 'APIStatus' },
      { icon: Cpu, label: 'GPU / Ollama Status', sublabel: 'Online · RTX 4080', target: 'GPUStatus' },
      { icon: History, label: 'Audit Logs', sublabel: 'View session history', target: 'AuditLogs' },
    ],
  },
];

export default function SettingsScreen() {
  const navigation = useNavigation<any>();
  const { theme, isDark, toggleTheme } = useTheme();
  const s = styles(theme);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <View style={s.header}>
        <Text style={s.title}>Admin & Settings</Text>
        <Text style={s.subtitle}>System configuration & preferences</Text>
      </View>

      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>

        {/* ─── THEME TOGGLE ─── */}
        <View style={s.section}>
          <Text style={s.sectionTitle}>Appearance</Text>
          <View style={s.sectionCard}>
            <TouchableOpacity style={s.settingRow} onPress={toggleTheme} activeOpacity={0.7}>
              <View style={s.settingIcon}>
                {isDark ? <Moon color={theme.primary} size={18} /> : <Sun color={theme.primary} size={18} />}
              </View>
              <View style={s.settingText}>
                <Text style={s.settingLabel}>{isDark ? 'Dark Mode' : 'Light Mode'}</Text>
                <Text style={s.settingSublabel}>Tap to switch to {isDark ? 'light' : 'dark'} mode</Text>
              </View>
              {/* Visual toggle pill */}
              <View style={[s.togglePill, isDark ? s.togglePillDark : s.togglePillLight]}>
                <View style={[s.toggleDot, isDark ? s.toggleDotRight : s.toggleDotLeft]} />
              </View>
            </TouchableOpacity>
          </View>
        </View>

        {settingsSections.map((section) => (
          <View key={section.title} style={s.section}>
            <Text style={s.sectionTitle}>{section.title}</Text>
            <View style={s.sectionCard}>
              {section.items.map((item, index) => (
                <TouchableOpacity
                  key={item.label}
                  style={[s.settingRow, index < section.items.length - 1 && s.settingRowBorder]}
                  activeOpacity={0.7}
                  onPress={() => {
                    if (item.target === 'Profile') {
                      navigation.navigate('Profile');
                    } else if (item.target === 'Security') {
                      navigation.navigate('Security');
                    } else {
                      // Placeholder for other screens
                      alert(`${item.label} coming soon in this demo.`);
                    }
                  }}
                >
                  <View style={s.settingIcon}>
                    <item.icon color={theme.primary} size={18} />
                  </View>
                  <View style={s.settingText}>
                    <Text style={s.settingLabel}>{item.label}</Text>
                    <Text style={s.settingSublabel}>{item.sublabel}</Text>
                  </View>
                  <ChevronRight color={theme.mutedForeground} size={18} />
                </TouchableOpacity>
              ))}
            </View>
          </View>
        ))}

        <TouchableOpacity style={s.logoutButton} onPress={() => navigation.replace('Splash')} activeOpacity={0.85}>
          <LogOut color={theme.destructive} size={18} style={{ marginRight: 8 }} />
          <Text style={s.logoutText}>End Session & Sign Out</Text>
        </TouchableOpacity>

        <Text style={s.versionInfo}>XMedFusion Mobile v1.0.0 · SDK 54{'\n'}Clinical Build — Confidential</Text>
      </ScrollView>
    </View>
  );
}

const styles = (theme: ReturnType<typeof useTheme>['theme']) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.background },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.xxl, paddingBottom: spacing.md },
  title: { color: theme.foreground, fontSize: typography['2xl'], fontWeight: '700', fontFamily: fontFamily.bold },
  subtitle: { color: theme.mutedForeground, fontSize: typography.sm, marginTop: 2, fontFamily: fontFamily.regular },
  scroll: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl },
  section: { marginBottom: spacing.lg },
  sectionTitle: { color: theme.mutedForeground, fontSize: typography.xs, fontWeight: '600', letterSpacing: 1, textTransform: 'uppercase', marginBottom: spacing.sm, marginLeft: 4, fontFamily: fontFamily.semiBold },
  sectionCard: { backgroundColor: theme.card, borderWidth: 1, borderColor: theme.cardBorder, borderRadius: radius.lg, overflow: 'hidden' },
  settingRow: { flexDirection: 'row', alignItems: 'center', padding: spacing.md, gap: spacing.md },
  settingRowBorder: { borderBottomWidth: 1, borderBottomColor: theme.cardBorder },
  settingIcon: { width: 36, height: 36, borderRadius: radius.sm, backgroundColor: theme.primaryGlow, alignItems: 'center', justifyContent: 'center' },
  settingText: { flex: 1 },
  settingLabel: { color: theme.foreground, fontSize: typography.base, fontWeight: '500', fontFamily: fontFamily.medium },
  settingSublabel: { color: theme.mutedForeground, fontSize: typography.xs, marginTop: 2, fontFamily: fontFamily.regular },
  chevron: { color: theme.mutedForeground, fontSize: 20 },
  // Toggle switch
  togglePill: { width: 46, height: 26, borderRadius: 13, justifyContent: 'center', paddingHorizontal: 3 },
  togglePillDark: { backgroundColor: theme.primary },
  togglePillLight: { backgroundColor: theme.cardBorder },
  toggleDot: { width: 20, height: 20, borderRadius: 10, backgroundColor: theme.white },
  toggleDotRight: { alignSelf: 'flex-end' },
  toggleDotLeft: { alignSelf: 'flex-start' },
  logoutButton: { 
    flexDirection: 'row', 
    height: 52, 
    borderWidth: 1, 
    borderColor: theme.destructive, 
    borderRadius: radius.lg, 
    alignItems: 'center', 
    justifyContent: 'center', 
    backgroundColor: theme.destructiveBg, 
    marginBottom: spacing.xl 
  },
  logoutText: { color: theme.destructive, fontWeight: '600', fontSize: typography.base, fontFamily: fontFamily.semiBold },
  versionInfo: { color: theme.mutedForeground, fontSize: typography.xs, textAlign: 'center', lineHeight: 18, fontFamily: fontFamily.regular },
});
