import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  ScrollView,
} from 'react-native';
import { 
  Info, 
  Dna, 
  Search, 
  Heart,
  Circle,
  Network
} from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';

const nodes = [
  { id: 1, label: 'Pneumonia', type: 'disease', connections: ['Consolidation', 'Fever', 'Infiltrates'] },
  { id: 2, label: 'Cardiomegaly', type: 'disease', connections: ['Heart Failure', 'Hypertension'] },
  { id: 3, label: 'Atelectasis', type: 'disease', connections: ['Consolidation', 'Pleural Effusion'] },
  { id: 4, label: 'Pleural Effusion', type: 'finding', connections: ['Atelectasis', 'Pneumonia'] },
];

export default function KnowledgeGraphScreen() {
  const { theme, isDark } = useTheme();
  const [selected, setSelected] = useState<typeof nodes[0] | null>(null);

  const typeColors: Record<string, string> = {
    disease: theme.destructive,
    finding: theme.warning,
    anatomy: theme.accent,
  };

  const typeBg: Record<string, string> = {
    disease: theme.destructiveBg,
    finding: theme.warningBg,
    anatomy: 'rgba(56, 100, 220, 0.12)',
  };

  const getNodeIcon = (type: string, color: string) => {
    switch (type) {
      case 'disease': return <Dna color={color} size={22} />;
      case 'finding': return <Search color={color} size={22} />;
      default: return <Heart color={color} size={22} />;
    }
  };

  const s = styles(theme);

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={theme.backgroundDeep} />

      {/* Header */}
      <View style={s.header}>
        <Text style={s.title}>Knowledge Graph</Text>
        <Text style={s.subtitle}>Clinical concept relationships</Text>
      </View>

      {/* Graph Visualization (simplified node list for mobile) */}
      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        {/* Info Banner */}
        <View style={s.infoBanner}>
          <Info color={theme.primary} size={18} />
          <Text style={s.infoBannerText}>
             Tap a node to explore its clinical relationships
          </Text>
        </View>

        {/* Nodes */}
        <View style={s.nodesGrid}>
          {nodes.map((node) => (
            <TouchableOpacity
              key={node.id}
              style={[
                s.nodeCard,
                selected?.id === node.id && s.nodeCardSelected,
              ]}
              onPress={() => setSelected(selected?.id === node.id ? null : node)}
              activeOpacity={0.8}
            >
              <View style={[s.nodeIcon, { backgroundColor: typeBg[node.type] }]}>
                {getNodeIcon(node.type, typeColors[node.type])}
              </View>
              <View style={s.nodeInfo}>
                <Text style={s.nodeLabel}>{node.label}</Text>
                <View style={[s.typeBadge, { backgroundColor: typeBg[node.type] }]}>
                  <Text style={[s.typeText, { color: typeColors[node.type] }]}>{node.type}</Text>
                </View>
              </View>
              <View style={s.connBadge}>
                <Network color={theme.mutedForeground} size={12} />
                <Text style={s.connCount}>{node.connections.length}</Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>

        {/* Expanded node detail */}
        {selected && (
          <View style={s.detailCard}>
            <Text style={s.detailTitle}>Relationships: {selected.label}</Text>
            {selected.connections.map((conn) => (
              <View key={conn} style={s.connectionRow}>
                <Circle color={theme.primary} size={8} fill={theme.primary} />
                <Text style={s.connText}>{conn}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Stats */}
        <View style={s.statsRow}>
          {[
            { label: 'Total Nodes', value: '148' },
            { label: 'Relationships', value: '412' },
            { label: 'Diseases', value: '67' },
          ].map((stat) => (
            <View key={stat.label} style={s.statCard}>
              <Text style={s.statValue}>{stat.value}</Text>
              <Text style={s.statLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.background },
  header: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.xxl,
    paddingBottom: spacing.md,
  },
  title: { color: theme.foreground, fontSize: typography['2xl'], fontWeight: '700', fontFamily: fontFamily.bold },
  subtitle: { color: theme.mutedForeground, fontSize: typography.sm, marginTop: 2, fontFamily: fontFamily.regular },
  scroll: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl, gap: spacing.md },
  infoBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.primaryGlow,
    borderWidth: 1,
    borderColor: theme.primary,
    borderRadius: radius.md,
    padding: spacing.md,
    gap: spacing.sm,
  },
  infoBannerText: { flex: 1, color: theme.primary, fontSize: typography.sm, fontFamily: fontFamily.medium },
  nodesGrid: { gap: spacing.sm },
  nodeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.cardBorder,
    borderRadius: radius.md,
    padding: spacing.md,
    gap: spacing.md,
  },
  nodeCardSelected: {
    borderColor: theme.primary,
    backgroundColor: theme.primaryGlow,
  },
  nodeIcon: {
    width: 44,
    height: 44,
    borderRadius: radius.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  nodeInfo: { flex: 1, gap: 4 },
  nodeLabel: { color: theme.foreground, fontWeight: '600', fontSize: typography.base, fontFamily: fontFamily.semiBold },
  typeBadge: { alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: radius.full },
  typeText: { fontSize: typography.xs, fontWeight: '600', textTransform: 'capitalize', fontFamily: fontFamily.semiBold },
  connBadge: { flexDirection: 'row', alignItems: 'center', gap: 4, opacity: 0.7 },
  connCount: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  detailCard: {
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.primary,
    borderRadius: radius.lg,
    padding: spacing.lg,
    gap: spacing.sm,
  },
  detailTitle: { color: theme.primary, fontWeight: '700', fontSize: typography.base, marginBottom: spacing.xs, fontFamily: fontFamily.bold },
  connectionRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, paddingVertical: 4 },
  connText: { color: theme.foreground, fontSize: typography.sm, fontFamily: fontFamily.regular },
  statsRow: { flexDirection: 'row', gap: spacing.sm },
  statCard: {
    flex: 1,
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.cardBorder,
    borderRadius: radius.md,
    padding: spacing.md,
    alignItems: 'center',
  },
  statValue: { color: theme.primary, fontSize: typography.xl, fontWeight: '700', fontFamily: fontFamily.bold },
  statLabel: { color: theme.mutedForeground, fontSize: typography.xs, textAlign: 'center', marginTop: 2, fontFamily: fontFamily.regular },
});
