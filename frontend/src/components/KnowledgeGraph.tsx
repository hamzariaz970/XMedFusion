import { useState, useCallback, useMemo } from "react";
// REMOVE Layout import
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { 
  Network, 
  Search, 
  ZoomIn, 
  ZoomOut, 
  Maximize2,
  Info,
  X,
  Layers,
  Activity,
  AlertCircle,
  FileText,
  MapPin,    
  Link as LinkIcon
} from "lucide-react";
import { cn } from "@/lib/utils";

// ... [Keep DATA TYPES, NODE COLORS, RELATION_TYPES, STATIC DATA, and STATIC_LINKS exactly as they are] ...
// ... [Keep calculateLayout logic exactly as it is] ...

// ------------------------------------------------------------------
// 1. DATA TYPES
// ------------------------------------------------------------------

export interface KnowledgeGraphProps {
  data?: {
    entities: [string, string][]; 
    relations: [number, number, string][]; 
  } | null;
}

interface Node {
  id: string | number;
  label: string;
  type: string;
  x: number;
  y: number;
  description?: string;
}

// Node Colors
const nodeTypeColors: Record<string, { bg: string; border: string; text: string; fill: string; icon: any }> = {
  modality: { bg: "bg-primary/20", border: "border-primary", text: "text-primary", fill: "fill-primary/20", icon: FileText },
  anatomy: { bg: "bg-blue-500/20", border: "border-blue-500", text: "text-blue-600 dark:text-blue-400", fill: "fill-blue-500/20", icon: Layers },
  finding: { bg: "bg-emerald-500/20", border: "border-emerald-500", text: "text-emerald-600 dark:text-emerald-400", fill: "fill-emerald-500/20", icon: Activity },
  diagnosis: { bg: "bg-amber-500/20", border: "border-amber-500", text: "text-amber-600 dark:text-amber-400", fill: "fill-amber-500/20", icon: AlertCircle },
  uncertain: { bg: "bg-muted/20", border: "border-muted-foreground", text: "text-muted-foreground", fill: "fill-muted-foreground/20", icon: Info },
};

// Relation Types Legend
const RELATION_TYPES = [
  { label: "located_at", description: "Finding in location", icon: MapPin, color: "text-blue-500" },
  { label: "suggestive_of", description: "Implies diagnosis", icon: Activity, color: "text-amber-500" },
  { label: "modify", description: "Descriptor", icon: LinkIcon, color: "text-gray-500" }
];

// ------------------------------------------------------------------
// 2. STATIC DATA (Aligned Tree Structure)
// ------------------------------------------------------------------
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
  { source: "chest_xray", target: "lungs" },
  { source: "chest_xray", target: "heart" },
  { source: "chest_xray", target: "mediastinum" },
  { source: "lungs", target: "lung_normal" },
  { source: "lungs", target: "consolidation" },
  { source: "lungs", target: "pleural_effusion" },
  { source: "heart", target: "cardiomegaly" },
  { source: "heart", target: "heart_normal" },
  { source: "mediastinum", target: "widened" },
  { source: "consolidation", target: "pneumonia" },
  { source: "pleural_effusion", target: "heart_failure" },
  { source: "cardiomegaly", target: "heart_failure" },
  { source: "lung_normal", target: "normal" },
  { source: "heart_normal", target: "normal" },
];

// ------------------------------------------------------------------
// 3. LAYOUT LOGIC
// ------------------------------------------------------------------
const calculateLayout = (entities: [string, string][]) => {
  const width = 800;
  
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
    const node = { id: index, label: text, type, description };

    if (type === "anatomy") anatomyNodes.push(node);
    else if (type === "diagnosis") diagnosisNodes.push(node);
    else observationNodes.push(node);
  });

  const nodes: Node[] = [];
  const anatomyY = 150;
  anatomyNodes.forEach((node, i) => {
    const step = width / (anatomyNodes.length + 1);
    nodes.push({ ...node, x: step * (i + 1), y: anatomyY });
  });

  const findingY = 320;
  observationNodes.forEach((node, i) => {
    const step = width / (observationNodes.length + 1);
    nodes.push({ ...node, x: step * (i + 1), y: findingY });
  });

  const diagnosisY = 480;
  diagnosisNodes.forEach((node, i) => {
    const step = width / (diagnosisNodes.length + 1);
    nodes.push({ ...node, x: step * (i + 1), y: diagnosisY });
  });

  return nodes;
};

// ------------------------------------------------------------------
// 4. MAIN COMPONENT (Cleaned Up)
// ------------------------------------------------------------------
const KnowledgeGraph = ({ data }: KnowledgeGraphProps) => {
  const [selectedNodeId, setSelectedNodeId] = useState<string | number | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [zoom, setZoom] = useState(1);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | number | null>(null);

  const { nodes, links } = useMemo(() => {
    if (data && data.entities && data.entities.length > 0) {
      const computedNodes = calculateLayout(data.entities);
      const computedLinks = data.relations.map(([src, tgt, type]) => ({
        source: src,
        target: tgt,
        label: type
      }));
      return { nodes: computedNodes, links: computedLinks };
    } else {
      return { nodes: STATIC_NODES, links: STATIC_LINKS };
    }
  }, [data]);

  const filteredNodes = useMemo(() => {
    if (!searchQuery) return nodes;
    return nodes.filter(node => 
      node.label.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [searchQuery, nodes]);

  const getNodePosition = useCallback((id: string | number) => {
    const node = nodes.find(n => n.id === id);
    return node ? { x: node.x, y: node.y } : { x: 0, y: 0 };
  }, [nodes]);

  const isNodeHighlighted = useCallback((id: string | number) => {
    if (hoveredNodeId === null && selectedNodeId === null) return true;
    const targetId = hoveredNodeId ?? selectedNodeId;
    if (id === targetId) return true;
    return links.some(link => 
      (link.source === targetId && link.target === id) || 
      (link.target === targetId && link.source === id)
    );
  }, [hoveredNodeId, selectedNodeId, links]);

  const isLinkHighlighted = useCallback((source: string | number, target: string | number) => {
    if (hoveredNodeId === null && selectedNodeId === null) return true;
    const targetId = hoveredNodeId ?? selectedNodeId;
    return source === targetId || target === targetId;
  }, [hoveredNodeId, selectedNodeId]);

  // REMOVED <Layout>, <section>, and inner titles. 
  // Just returning the Grid structure now.
  return (
    <div className="flex flex-col gap-6"> 
      <div className="grid lg:grid-cols-12 gap-6">
        
        {/* LEFT SIDEBAR (3 Cols) */}
        <div className="lg:col-span-3 space-y-4">
          {/* 1. Search */}
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm">Search Nodes</CardTitle></CardHeader>
            <CardContent>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input placeholder="Search concepts..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="pl-9" />
              </div>
            </CardContent>
          </Card>

          {/* 2. Node Types Legend */}
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm">Node Legend</CardTitle></CardHeader>
            <CardContent className="space-y-2">
              {Object.entries(nodeTypeColors).map(([type, colors]) => (
                <div key={type} className="flex items-center gap-2">
                  <div className={cn("w-4 h-4 rounded-full border-2", colors.bg, colors.border)} />
                  <span className="text-sm capitalize text-foreground">{type}</span>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* 3. Relation Types Legend */}
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm">Relation Types</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              {RELATION_TYPES.map((rel) => (
                <div key={rel.label} className="flex items-start gap-2">
                  <rel.icon className={`w-3 h-3 mt-0.5 ${rel.color}`} />
                  <div>
                    <div className="text-xs font-bold">{rel.label}</div>
                    <div className="text-[10px] text-muted-foreground leading-tight">{rel.description}</div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* 4. Controls */}
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm">View Controls</CardTitle></CardHeader>
            <CardContent className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => setZoom(z => Math.max(0.5, z - 0.1))}><ZoomOut className="w-4 h-4" /></Button>
              <Button variant="outline" size="sm" onClick={() => setZoom(z => Math.min(1.5, z + 0.1))}><ZoomIn className="w-4 h-4" /></Button>
              <Button variant="outline" size="sm" onClick={() => setZoom(1)}><Maximize2 className="w-4 h-4" /></Button>
            </CardContent>
          </Card>

          {selectedNodeId !== null && (
            <Card className="border-primary/30 animate-in fade-in slide-in-from-bottom-2">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Info className="w-4 h-4 text-primary" /> Details
                  </CardTitle>
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setSelectedNodeId(null)}><X className="w-3 h-3" /></Button>
                </div>
              </CardHeader>
              <CardContent className="pt-4">
                {(() => {
                  const node = nodes.find(n => n.id === selectedNodeId);
                  if (!node) return null;
                  const colors = nodeTypeColors[node.type] || nodeTypeColors.uncertain;
                  return (
                    <div className="space-y-3">
                      <Badge className={cn(colors.bg, colors.text, "border", colors.border)}>{node.type.toUpperCase()}</Badge>
                      <h4 className="font-bold text-lg">{node.label}</h4>
                      <p className="text-sm text-muted-foreground">{node.description || "No description available."}</p>
                    </div>
                  );
                })()}
              </CardContent>
            </Card>
          )}
        </div>

        {/* MAIN GRAPH (9 Cols) */}
        <Card className="lg:col-span-9 overflow-hidden border-2 shadow-inner bg-slate-50/50 dark:bg-slate-950/50">
          <CardHeader className="border-b px-4 py-3 bg-background/50 backdrop-blur-sm sticky top-0 z-10">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-base"><Network className="w-5 h-5 text-primary" /> Visualization</CardTitle>
              <Badge variant="secondary" className="font-normal">{nodes.length} Nodes • {links.length} Links</Badge>
            </div>
          </CardHeader>
          <CardContent className="p-0 relative">
            <div className="relative w-full h-[600px] overflow-hidden cursor-move bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px] dark:bg-[radial-gradient(#1f2937_1px,transparent_1px)]">
              <svg width="100%" height="100%" viewBox="0 0 800 600" style={{ transform: `scale(${zoom})`, transformOrigin: 'center', transition: 'transform 0.3s ease-out' }}>
                {links.map((link, index) => {
                  const sourcePos = getNodePosition(link.source);
                  const targetPos = getNodePosition(link.target);
                  const highlighted = isLinkHighlighted(link.source, link.target);
                  return (
                    <g key={index}>
                      <line x1={sourcePos.x} y1={sourcePos.y} x2={targetPos.x} y2={targetPos.y} stroke={highlighted ? "hsl(var(--primary))" : "hsl(var(--border))"} strokeWidth={highlighted ? 2 : 1.5} strokeOpacity={highlighted ? 0.6 : 0.2} className="transition-all duration-300" />
                      {highlighted && link.label && (
                        <text x={(sourcePos.x + targetPos.x) / 2} y={(sourcePos.y + targetPos.y) / 2} textAnchor="middle" fill="hsl(var(--muted-foreground))" fontSize="10" dy={-5} className="bg-background/80 px-1 rounded">{link.label}</text>
                      )}
                    </g>
                  );
                })}
                {filteredNodes.map((node) => {
                  const colors = nodeTypeColors[node.type] || nodeTypeColors.uncertain;
                  const highlighted = isNodeHighlighted(node.id);
                  const isSelected = selectedNodeId === node.id;
                  const Icon = colors.icon;
                  return (
                    <g key={node.id} transform={`translate(${node.x}, ${node.y})`} onClick={() => setSelectedNodeId(node.id)} onMouseEnter={() => setHoveredNodeId(node.id)} onMouseLeave={() => setHoveredNodeId(null)} className="cursor-pointer transition-opacity duration-300" style={{ opacity: highlighted ? 1 : 0.2 }}>
                      <circle r={isSelected ? 30 : 26} fill="hsl(var(--card))" stroke={isSelected ? "hsl(var(--primary))" : "hsl(var(--border))"} strokeWidth={isSelected ? 3 : 2} className="transition-all duration-200" />
                      <circle r={18} className={cn(colors.fill)} />
                      <foreignObject x="-9" y="-9" width="18" height="18" className="pointer-events-none"><div className={cn("w-full h-full flex items-center justify-center", colors.text)}><Icon size={14} /></div></foreignObject>
                      <text y={42} textAnchor="middle" fontSize="11" fontWeight={isSelected ? "700" : "500"} fill="hsl(var(--foreground))" className="pointer-events-none drop-shadow-sm">{node.label.length > 15 ? node.label.substring(0, 15) + '...' : node.label}</text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* STATS BAR (Separate grid) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
        {[ { label: "Total Entities", value: nodes.length }, { label: "Anatomical Sites", value: nodes.filter(n => n.type === 'anatomy').length }, { label: "Observations", value: nodes.filter(n => n.type === 'finding' || n.type === 'diagnosis').length }, { label: "Connections", value: links.length } ].map((stat, index) => (
          <Card key={index} className="bg-muted/10"><CardContent className="p-4 text-center"><div className="text-2xl font-bold text-primary">{stat.value}</div><div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{stat.label}</div></CardContent></Card>
        ))}
      </div>
    </div>
  );
};

export default KnowledgeGraph;