import React, { useState, useMemo } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, ScrollView, Dimensions } from 'react-native';
import Svg, { Circle, Line, Text as SvgText, G } from 'react-native-svg';
import { Info, Layers, Activity, AlertCircle, FileText, Share } from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, radius, typography, fontFamily } from '../theme/colors';
import { useReportStore } from '../store/reportStore';

const { width: scrWidth } = Dimensions.get('window');
const graphWidth = 800;
const graphHeight = 600;

export const nodeTypeColors = (theme: any) => ({
  modality: { bg: theme.primary, text: theme.white, border: theme.primary, fill: theme.primary + '30' },
  anatomy: { bg: '#3b82f6', text: theme.white, border: '#2563eb', fill: '#3b82f630' },
  finding: { bg: '#10b981', text: theme.white, border: '#059669', fill: '#10b98130' },
  diagnosis: { bg: '#f59e0b', text: theme.white, border: '#d97706', fill: '#f59e0b30' },
  uncertain: { bg: theme.mutedForeground, text: theme.white, border: theme.mutedForeground, fill: theme.mutedForeground + '30' },
});

const STATIC_NODES = [
  { id: "chest_xray", label: "Chest X-ray", type: "modality", x: 400, y: 60, description: "Primary imaging modality" },
  { id: "lungs", label: "Lungs", type: "anatomy", x: 200, y: 200, description: "Bilateral lung fields" },
  { id: "heart", label: "Heart", type: "anatomy", x: 400, y: 200, description: "Cardiac silhouette" },
  { id: "mediastinum", label: "Mediastinum", type: "anatomy", x: 600, y: 200, description: "Central compartment" },
  { id: "lung_normal", label: "Clear Fields", type: "finding", x: 100, y: 380, description: "No infiltrates" },
  { id: "consolidation", label: "Consolidation", type: "finding", x: 200, y: 380, description: "Airspace opacity" },
  { id: "pleural_effusion", label: "Pleural Effusion", type: "finding", x: 300, y: 380, description: "Fluid accumulation" },
  { id: "cardiomegaly", label: "Cardiomegaly", type: "finding", x: 450, y: 380, description: "Enlarged heart" },
  { id: "heart_normal", label: "Normal Size", type: "finding", x: 550, y: 380, description: "Normal limits" },
  { id: "widened", label: "Widening", type: "finding", x: 650, y: 380, description: "Mediastinal widening" },
  { id: "pneumonia", label: "Pneumonia", type: "diagnosis", x: 200, y: 520, description: "Infection" },
  { id: "heart_failure", label: "Heart Failure", type: "diagnosis", x: 450, y: 520, description: "Decompensation" },
  { id: "normal", label: "Normal Study", type: "diagnosis", x: 650, y: 520, description: "No abnormalities" },
];

const STATIC_LINKS = [
  { source: "chest_xray", target: "lungs", label: "visualizes" },
  { source: "chest_xray", target: "heart", label: "visualizes" },
  { source: "chest_xray", target: "mediastinum", label: "visualizes" },
  { source: "lungs", target: "lung_normal", label: "has_finding" },
  { source: "lungs", target: "consolidation", label: "has_finding" },
  { source: "lungs", target: "pleural_effusion", label: "has_finding" },
  { source: "heart", target: "cardiomegaly", label: "has_finding" },
  { source: "heart", target: "heart_normal", label: "has_finding" },
  { source: "mediastinum", target: "widened", label: "has_finding" },
  { source: "consolidation", target: "pneumonia", label: "suggestive_of" },
  { source: "pleural_effusion", target: "heart_failure", label: "suggestive_of" },
  { source: "cardiomegaly", target: "heart_failure", label: "suggestive_of" },
  { source: "lung_normal", target: "normal", label: "suggestive_of" },
  { source: "heart_normal", target: "normal", label: "suggestive_of" },
];

export const calculateLayout = (entities: [string, string][], width: number = 800) => {
  const anatomyNodes: any[] = [];
  const observationNodes: any[] = [];
  const diagnosisNodes: any[] = [];

  entities.forEach(([text, label], index) => {
    let type = "uncertain";
    if (label.includes("Anatomy")) type = "anatomy";
    else if (label.includes("Observation")) type = "finding";

    if ((text.toLowerCase().includes("normal") || text.toLowerCase().includes("pneumonia") || text.toLowerCase().includes("edema")) && type === "finding") {
      type = "diagnosis";
    }

    const description = label.split("::")[1]?.replace("_", " ") || label;
    const node = { id: index.toString(), label: text, type, description };

    if (type === "anatomy") anatomyNodes.push(node);
    else if (type === "diagnosis") diagnosisNodes.push(node);
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

export default function KnowledgeGraphScreen() {
  const { theme, isDark } = useTheme();
  const reports = useReportStore((state) => state.reports);
  const [selectedReportIndex, setSelectedReportIndex] = useState(0);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const activeReport = reports.length > 0 ? reports[selectedReportIndex] : null;

  const { nodes, links } = useMemo(() => {
    if (activeReport && activeReport.knowledgeGraph && activeReport.knowledgeGraph.entities) {
      const kg = activeReport.knowledgeGraph;
      const computedNodes = calculateLayout(kg.entities);
      const computedLinks = kg.relations.map(([src, tgt, type]: [number, number, string]) => ({
        source: src.toString(),
        target: tgt.toString(),
        label: type
      }));
      return { nodes: computedNodes, links: computedLinks };
    }
    return { nodes: STATIC_NODES, links: STATIC_LINKS };
  }, [activeReport]);

  const getNodePosition = (id: string) => {
    const node = nodes.find(n => n.id === id);
    return node ? { x: node.x, y: node.y } : { x: 0, y: 0 };
  };

  const isHighlighted = (id: string) => {
    if (!selectedNodeId) return true;
    if (id === selectedNodeId) return true;
    return links.some((l: any) => (l.source === selectedNodeId && l.target === id) || (l.target === selectedNodeId && l.source === id));
  };

  const isLinkHighlighted = (source: string, target: string) => {
    if (!selectedNodeId) return true;
    return source === selectedNodeId || target === selectedNodeId;
  };

  const s = styles(theme);
  const colors = nodeTypeColors(theme);

  const selectedNodeData = selectedNodeId ? nodes.find(n => n.id === selectedNodeId) : null;

  return (
    <View style={s.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={theme.backgroundDeep} />

      <View style={s.header}>
        <Text style={s.title}>Knowledge Graphs</Text>
        <Text style={s.subtitle}>Medical concept relationships</Text>
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* History Selector */}
        {reports.length > 0 && (
          <View style={s.horizontalWrapper}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.historyRow}>
              {reports.map((rep, idx) => (
                <TouchableOpacity
                  key={rep.id}
                  style={[s.historyChip, selectedReportIndex === idx && s.historyChipActive]}
                  onPress={() => setSelectedReportIndex(idx)}
                >
                  <Text style={[s.historyChipText, selectedReportIndex === idx && s.historyChipTextActive]}>
                    {rep.patientId} • {new Date(rep.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        )}

        {/* Canvas Wrapper */}
        <View style={s.canvasWrapper}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <ScrollView showsVerticalScrollIndicator={false}>
              <View style={s.svgContainer}>
                <Svg width={graphWidth} height={graphHeight} viewBox={`0 0 ${graphWidth} ${graphHeight}`}>
                  {/* Links */}
                  {links.map((link: any, idx: number) => {
                    const srcPos = getNodePosition(link.source as string);
                    const tgtPos = getNodePosition(link.target as string);
                    const highlighted = isLinkHighlighted(link.source as string, link.target as string);
                    return (
                      <G key={`link-${idx}`}>
                        <Line
                          x1={srcPos.x} y1={srcPos.y}
                          x2={tgtPos.x} y2={tgtPos.y}
                          stroke={highlighted ? theme.primary : theme.border}
                          strokeWidth={highlighted ? 2 : 1.5}
                          opacity={highlighted ? 0.6 : 0.2}
                        />
                        {highlighted && link.label && (
                          <SvgText
                            x={(srcPos.x + tgtPos.x) / 2}
                            y={(srcPos.y + tgtPos.y) / 2 - 5}
                            fill={theme.mutedForeground}
                            fontSize={10}
                            textAnchor="middle"
                            fontFamily={fontFamily.regular}
                          >
                            {link.label}
                          </SvgText>
                        )}
                      </G>
                    );
                  })}

                  {/* Nodes */}
                  {nodes.map((node) => {
                    const highlighted = isHighlighted(node.id as string);
                    const isSelected = selectedNodeId === node.id;
                    const tColor = colors[node.type as keyof typeof colors] || colors.uncertain;

                    return (
                      <G
                        key={`node-${node.id}`}
                        x={node.x} y={node.y}
                        onPress={() => setSelectedNodeId(isSelected ? null : node.id as string)}
                        opacity={highlighted ? 1 : 0.2}
                      >
                        <Circle
                          r={isSelected ? 30 : 26}
                          fill={theme.card}
                          stroke={isSelected ? theme.primary : theme.border}
                          strokeWidth={isSelected ? 3 : 2}
                        />
                        <Circle r={18} fill={tColor.fill} />
                        <SvgText
                          y={42}
                          fill={theme.foreground}
                          fontSize={11}
                          fontWeight={isSelected ? '700' : '500'}
                          textAnchor="middle"
                          fontFamily={isSelected ? fontFamily.bold : fontFamily.semiBold}
                        >
                          {node.label.length > 15 ? node.label.substring(0, 15) + '...' : node.label}
                        </SvgText>
                      </G>
                    );
                  })}
                </Svg>
              </View>
            </ScrollView>
          </ScrollView>
        </View>

        {/* Selected Node Details */}
        {selectedNodeData && (
          <View style={s.detailCard}>
            <View style={s.detailHeader}>
              <Text style={s.detailTitle}>{selectedNodeData.label}</Text>
              <View style={[s.typeBadge, { backgroundColor: (colors[selectedNodeData.type as keyof typeof colors] || colors.uncertain).bg }]}>
                <Text style={s.typeText}>{selectedNodeData.type}</Text>
              </View>
            </View>
            <Text style={s.detailDesc}>{selectedNodeData.description || "No specific description available."}</Text>
          </View>
        )}

        {/* Stats Footer */}
        <View style={s.statsGrid}>
          {[
            { l: 'Entities', v: nodes.length },
            { l: 'Relations', v: links.length },
            { l: 'Anatomy', v: nodes.filter(n => n.type === 'anatomy').length },
            { l: 'Findings', v: nodes.filter(n => n.type === 'finding' || n.type === 'diagnosis').length }
          ].map((st, i) => (
            <View key={i} style={s.statBox}>
              <Text style={s.statVal}>{st.v}</Text>
              <Text style={s.statLab}>{st.l}</Text>
            </View>
          ))}
        </View>

      </ScrollView>
    </View>
  );
}

const styles = (theme: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.background },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.xxl, paddingBottom: spacing.sm },
  title: { color: theme.foreground, fontSize: typography['2xl'], fontWeight: '700', fontFamily: fontFamily.bold },
  subtitle: { color: theme.mutedForeground, fontSize: typography.sm, marginTop: 2, fontFamily: fontFamily.regular },
  horizontalWrapper: { paddingVertical: spacing.md },
  historyRow: { paddingHorizontal: spacing.lg, gap: spacing.sm },
  historyChip: { paddingHorizontal: 16, paddingVertical: 8, borderRadius: radius.full, backgroundColor: theme.card, borderWidth: 1, borderColor: theme.cardBorder },
  historyChipActive: { backgroundColor: theme.primaryGlow, borderColor: theme.primary },
  historyChipText: { color: theme.mutedForeground, fontSize: typography.sm, fontWeight: '600', fontFamily: fontFamily.semiBold },
  historyChipTextActive: { color: theme.primary },
  canvasWrapper: { height: 400, backgroundColor: theme.backgroundDeep, borderTopWidth: 1, borderBottomWidth: 1, borderColor: theme.border, overflow: 'hidden' },
  svgContainer: { width: graphWidth, height: graphHeight, backgroundColor: theme.background },
  detailCard: { margin: spacing.lg, backgroundColor: theme.card, borderRadius: radius.lg, padding: spacing.lg, borderWidth: 1, borderColor: theme.primary, shadowColor: theme.primary, shadowOpacity: 0.1, shadowRadius: 10 },
  detailHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.sm },
  detailTitle: { flex: 1, color: theme.foreground, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold },
  typeBadge: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: radius.sm },
  typeText: { color: '#fff', fontSize: 10, fontWeight: '700', textTransform: 'uppercase' },
  detailDesc: { color: theme.mutedForeground, fontSize: typography.sm, lineHeight: 20 },
  statsGrid: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: spacing.md, paddingBottom: spacing.xxl },
  statBox: { width: '25%', alignItems: 'center', paddingVertical: spacing.md },
  statVal: { color: theme.primary, fontSize: typography.lg, fontWeight: '700', fontFamily: fontFamily.bold },
  statLab: { color: theme.mutedForeground, fontSize: 10, textTransform: 'uppercase', marginTop: 4, fontFamily: fontFamily.semiBold },
});
