import React, { useMemo, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, ScrollView, useWindowDimensions } from 'react-native';
import Svg, { Circle, G, Line, Text as SvgText } from 'react-native-svg';
import { Activity, FileText } from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { glassPanel, shellBackground, surfaceCard } from '../theme/ui';
import { useAnalysis } from '../context/AnalysisContext';
import { useReportStore } from '../store/reportStore';

const graphWidth = 800;
const graphHeight = 600;

export const nodeTypeColors = (theme: any) => ({
  modality: { bg: theme.primary, fill: `${theme.primary}30` },
  anatomy: { bg: '#3b82f6', fill: '#3b82f630' },
  finding: { bg: '#10b981', fill: '#10b98130' },
  diagnosis: { bg: '#f59e0b', fill: '#f59e0b30' },
  uncertain: { bg: theme.mutedForeground, fill: `${theme.mutedForeground}30` },
});

export const calculateLayout = (entities: [string, string][], width: number = 800) => {
  const anatomyNodes: any[] = [];
  const observationNodes: any[] = [];
  const diagnosisNodes: any[] = [];

  entities.forEach(([text, label], index) => {
    let type = 'uncertain';
    if (label.includes('Anatomy')) type = 'anatomy';
    else if (label.includes('Observation')) type = 'finding';

    if ((text.toLowerCase().includes('normal') || text.toLowerCase().includes('pneumonia') || text.toLowerCase().includes('edema')) && type === 'finding') {
      type = 'diagnosis';
    }

    const node = { id: index.toString(), label: text, type, description: label };
    if (type === 'anatomy') anatomyNodes.push(node);
    else if (type === 'diagnosis') diagnosisNodes.push(node);
    else observationNodes.push(node);
  });

  const layoutNodes: any[] = [];
  const placeRow = (arr: any[], yVal: number) => {
    arr.forEach((node, i) => {
      const step = width / (arr.length + 1);
      layoutNodes.push({ ...node, x: step * (i + 1), y: yVal });
    });
  };

  placeRow(anatomyNodes, 150);
  placeRow(observationNodes, 320);
  placeRow(diagnosisNodes, 480);
  return layoutNodes;
};

const STATIC_NODES = [
  { id: '0', label: 'Chest X-ray', type: 'modality', x: 400, y: 60, description: 'Primary imaging modality' },
  { id: '1', label: 'Lungs', type: 'anatomy', x: 200, y: 200, description: 'Bilateral lung fields' },
  { id: '2', label: 'Heart', type: 'anatomy', x: 400, y: 200, description: 'Cardiac silhouette' },
  { id: '3', label: 'Mediastinum', type: 'anatomy', x: 600, y: 200, description: 'Central compartment' },
];

const STATIC_LINKS = [
  { source: '0', target: '1', label: 'visualizes' },
  { source: '0', target: '2', label: 'visualizes' },
  { source: '0', target: '3', label: 'visualizes' },
];

export default function KnowledgeGraphScreen() {
  const { theme, isDark } = useTheme();
  const { width } = useWindowDimensions();
  const { knowledgeGraphData, report } = useAnalysis();
  const reports = useReportStore((state) => state.reports);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const s = styles(theme);

  const graphData = useMemo(() => {
    if (knowledgeGraphData?.entities?.length) {
      const nodes = calculateLayout(knowledgeGraphData.entities);
      const links = knowledgeGraphData.relations.map(([src, tgt, type]: [number, number, string]) => ({
        source: src.toString(),
        target: tgt.toString(),
        label: type,
      }));
      return { nodes, links };
    }

    const fallbackReport = reports[0];
    if (fallbackReport?.knowledgeGraph?.entities?.length) {
      const nodes = calculateLayout(fallbackReport.knowledgeGraph.entities);
      const links = fallbackReport.knowledgeGraph.relations.map(([src, tgt, type]: [number, number, string]) => ({
        source: src.toString(),
        target: tgt.toString(),
        label: type,
      }));
      return { nodes, links };
    }

    return { nodes: STATIC_NODES, links: STATIC_LINKS };
  }, [knowledgeGraphData, reports]);

  const colors = nodeTypeColors(theme);
  const selectedNodeData = selectedNodeId ? graphData.nodes.find((node: any) => node.id === selectedNodeId) : null;

  const getNodePosition = (id: string) => {
    const node = graphData.nodes.find((item: any) => item.id === id);
    return node ? { x: node.x, y: node.y } : { x: 0, y: 0 };
  };

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <ScrollView contentInsetAdjustmentBehavior="automatic" contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.heroCard}>
          <Text style={s.title}>Knowledge Graphs</Text>
          <Text style={s.subtitle}>Medical concept relationships derived from the current analysis.</Text>
        </View>

        <View style={[s.canvasWrapper, { minHeight: width > 420 ? 430 : 380 }]}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <ScrollView showsVerticalScrollIndicator={false}>
              <View style={s.svgContainer}>
                <Svg width={graphWidth} height={graphHeight} viewBox={`0 0 ${graphWidth} ${graphHeight}`}>
                  {graphData.links.map((link: any, idx: number) => {
                    const srcPos = getNodePosition(link.source);
                    const tgtPos = getNodePosition(link.target);
                    return (
                      <G key={`link-${idx}`}>
                        <Line x1={srcPos.x} y1={srcPos.y} x2={tgtPos.x} y2={tgtPos.y} stroke={theme.border} strokeWidth={1.5} opacity={0.4} />
                        <SvgText x={(srcPos.x + tgtPos.x) / 2} y={(srcPos.y + tgtPos.y) / 2 - 5} fill={theme.mutedForeground} fontSize={10} textAnchor="middle" fontFamily={fontFamily.regular}>
                          {link.label}
                        </SvgText>
                      </G>
                    );
                  })}

                  {graphData.nodes.map((node: any) => {
                    const isSelected = selectedNodeId === node.id;
                    const tone = colors[node.type as keyof typeof colors] || colors.uncertain;
                    return (
                      <G key={`node-${node.id}`} x={node.x} y={node.y} onPress={() => setSelectedNodeId(isSelected ? null : node.id)}>
                        <Circle r={isSelected ? 30 : 26} fill={theme.card} stroke={isSelected ? theme.primary : theme.border} strokeWidth={isSelected ? 3 : 2} />
                        <Circle r={18} fill={tone.fill} />
                        <SvgText y={42} fill={theme.foreground} fontSize={11} fontWeight={isSelected ? '700' : '500'} textAnchor="middle" fontFamily={fontFamily.semiBold}>
                          {node.label.length > 15 ? `${node.label.substring(0, 15)}...` : node.label}
                        </SvgText>
                      </G>
                    );
                  })}
                </Svg>
              </View>
            </ScrollView>
          </ScrollView>
        </View>

        {selectedNodeData ? (
          <View style={s.detailCard}>
            <Text style={s.detailTitle}>{selectedNodeData.label}</Text>
            <Text style={s.detailDesc}>{selectedNodeData.description || 'No specific description available.'}</Text>
          </View>
        ) : null}

        <View style={s.statsRow}>
          <View style={s.statCard}>
            <Activity color={theme.primary} size={18} />
            <Text style={s.statValue}>{graphData.nodes.length}</Text>
            <Text style={s.statLabel}>Entities</Text>
          </View>
          <View style={s.statCard}>
            <FileText color={theme.primary} size={18} />
            <Text style={s.statValue}>{graphData.links.length}</Text>
            <Text style={s.statLabel}>Relations</Text>
          </View>
        </View>

        {report ? (
          <View style={s.reportCard}>
            <Text style={s.reportTitle}>Current Report Context</Text>
            <Text style={s.reportText}>{report.findings}</Text>
          </View>
        ) : null}
      </ScrollView>
    </View>
  );
}

const styles = (theme: any) => StyleSheet.create({
  container: { flex: 1, ...shellBackground(theme) },
  scroll: { paddingHorizontal: spacing.lg, paddingTop: spacing.xl, paddingBottom: spacing.xxl + 84, gap: spacing.md },
  heroCard: { ...glassPanel(theme), padding: spacing.lg, gap: spacing.sm },
  title: { color: theme.foreground, fontSize: typography['2xl'], fontFamily: fontFamily.extraBold },
  subtitle: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
  canvasWrapper: { ...surfaceCard(theme), overflow: 'hidden' },
  svgContainer: { padding: spacing.md },
  detailCard: { ...surfaceCard(theme), padding: spacing.md, gap: spacing.xs },
  detailTitle: { color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.bold },
  detailDesc: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 20, fontFamily: fontFamily.regular },
  statsRow: { flexDirection: 'row', gap: spacing.sm },
  statCard: { flex: 1, ...surfaceCard(theme), padding: spacing.md, gap: 4 },
  statValue: { color: theme.foreground, fontSize: typography.lg, fontFamily: fontFamily.extraBold },
  statLabel: { color: theme.mutedForeground, fontSize: typography.xs, fontFamily: fontFamily.regular },
  reportCard: { ...surfaceCard(theme), padding: spacing.md, gap: spacing.sm },
  reportTitle: { color: theme.foreground, fontSize: typography.base, fontFamily: fontFamily.bold },
  reportText: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 22, fontFamily: fontFamily.regular },
});
