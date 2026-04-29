import { useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";
import { 
  FileImage, 
  Eye, 
  EyeOff,
  ChevronRight,
  Info,
  Layers
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Link } from "react-router-dom";
import { useAnalysis } from "@/context/AnalysisContext";

// Hardcoded image regions and sentence mappings
const imageMappings = [
  {
    id: 1,
    sentence: "Both lung fields are clear with no evidence of consolidation, masses, or pleural effusion.",
    region: "Lungs",
    coordinates: { x: 20, y: 25, width: 60, height: 40 },
    color: "bg-medical-success/30 border-medical-success",
    confidence: 0.94,
  },
  {
    id: 2,
    sentence: "Cardiac silhouette is within normal limits. No cardiomegaly observed.",
    region: "Heart",
    coordinates: { x: 35, y: 35, width: 30, height: 30 },
    color: "bg-accent/30 border-accent",
    confidence: 0.92,
  },
  {
    id: 3,
    sentence: "Mediastinal contours are unremarkable. Trachea is midline.",
    region: "Mediastinum",
    coordinates: { x: 40, y: 10, width: 20, height: 25 },
    color: "bg-primary/30 border-primary",
    confidence: 0.89,
  },
  {
    id: 4,
    sentence: "Visualized osseous structures show no acute abnormality.",
    region: "Bones",
    coordinates: { x: 10, y: 20, width: 15, height: 50 },
    color: "bg-medical-warning/30 border-medical-warning",
    confidence: 0.91,
  },
  {
    id: 5,
    sentence: "Both hemidiaphragms are well-defined with normal costophrenic angles.",
    region: "Diaphragm",
    coordinates: { x: 20, y: 60, width: 60, height: 15 },
    color: "bg-purple-500/30 border-purple-500",
    confidence: 0.93,
  },
];

const ImageMapping = () => {
  const [selectedMapping, setSelectedMapping] = useState<number | null>(null);
  const [showOverlays, setShowOverlays] = useState(true);
  const [hoveredMapping, setHoveredMapping] = useState<number | null>(null);
  const { previewUrl } = useAnalysis();

  return (
    <Layout>
      <section className="figma-page-shell">
        <div className="space-y-10">
          <div className="figma-workspace-hero grid w-full gap-6 lg:grid-cols-[1fr_380px] lg:items-center">
            <div>
              <Badge variant="outline" className="eyebrow mb-4">
                <Layers className="h-3.5 w-3.5" />
                Evidence localization
              </Badge>
              <h1 className="mb-4 text-3xl font-extrabold tracking-tight text-foreground md:text-5xl">
                Image-Report <span className="text-primary">Mapping</span>
              </h1>
              <p className="max-w-2xl text-muted-foreground">
                Visualize how report findings are mapped to specific regions of the X-ray image for evidence-based transparency.
              </p>
            </div>
            <RadiologyImageCard
              src={previewUrl || radiologyImages.chestXrayReview}
              alt="Radiologist reviewing report mapping"
              label="Region mapping"
              caption="Report sentences connected to scan areas"
              className="min-h-[240px]"
            />
          </div>

          <div className="grid w-full gap-8 lg:grid-cols-3">
            {/* Image Display */}
            <div className="lg:col-span-2 space-y-4">
              <Card className="surface-card overflow-hidden">
                <CardHeader className="border-b border-border/50 bg-secondary/40">
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <FileImage className="w-5 h-5 text-primary" />
                      X-ray Analysis View
                    </CardTitle>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowOverlays(!showOverlays)}
                    >
                      {showOverlays ? (
                        <>
                          <EyeOff className="w-4 h-4 mr-2" />
                          Hide Overlays
                        </>
                      ) : (
                        <>
                          <Eye className="w-4 h-4 mr-2" />
                          Show Overlays
                        </>
                      )}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="relative aspect-square bg-clinical-ink">
                    {previewUrl ? (
                      <img src={previewUrl} alt="Selected scan" className="absolute inset-0 h-full w-full object-contain p-4" />
                    ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <svg viewBox="0 0 400 400" className="w-full h-full opacity-30">
                        {/* Simplified X-ray representation */}
                        <ellipse cx="200" cy="200" rx="150" ry="180" fill="none" stroke="white" strokeWidth="2" />
                        <ellipse cx="200" cy="160" rx="60" ry="80" fill="none" stroke="white" strokeWidth="1.5" />
                        <path d="M140 100 Q200 80 260 100" fill="none" stroke="white" strokeWidth="1.5" />
                        <ellipse cx="140" cy="200" rx="50" ry="70" fill="none" stroke="white" strokeWidth="1" />
                        <ellipse cx="260" cy="200" rx="50" ry="70" fill="none" stroke="white" strokeWidth="1" />
                        <line x1="200" y1="80" x2="200" y2="150" stroke="white" strokeWidth="1.5" />
                        <path d="M100 320 Q200 350 300 320" fill="none" stroke="white" strokeWidth="1.5" />
                      </svg>
                    </div>
                    )}

                    {/* Region Overlays */}
                    {showOverlays && imageMappings.map((mapping) => (
                      <div
                        key={mapping.id}
                        className={cn(
                          "absolute cursor-pointer rounded-[18px] border-2 transition-all duration-300",
                          mapping.color,
                          (selectedMapping === mapping.id || hoveredMapping === mapping.id)
                            ? "opacity-100 scale-105"
                            : "opacity-60 hover:opacity-80"
                        )}
                        style={{
                          left: `${mapping.coordinates.x}%`,
                          top: `${mapping.coordinates.y}%`,
                          width: `${mapping.coordinates.width}%`,
                          height: `${mapping.coordinates.height}%`,
                        }}
                        onClick={() => setSelectedMapping(mapping.id)}
                        onMouseEnter={() => setHoveredMapping(mapping.id)}
                        onMouseLeave={() => setHoveredMapping(null)}
                      >
                        {(selectedMapping === mapping.id || hoveredMapping === mapping.id) && (
                          <div className="absolute -top-8 left-1/2 -translate-x-1/2 whitespace-nowrap">
                            <Badge variant="secondary" className="shadow-lg">
                              {mapping.region}
                            </Badge>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Controls */}
              <div className="flex items-center justify-between">
                <div className="flex gap-2">
                  <Badge variant="secondary" className="gap-1">
                    <Layers className="w-3 h-3" />
                    {imageMappings.length} Regions
                  </Badge>
                </div>
                <Link to="/upload">
                  <Button variant="outline" size="sm">
                    Upload New Image
                  </Button>
                </Link>
              </div>
            </div>

            {/* Findings List */}
            <div className="space-y-4">
              <Card className="surface-card">
                <CardHeader>
                  <CardTitle className="text-lg">Report Findings</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Click on a finding to highlight its corresponding region
                  </p>
                </CardHeader>
                <CardContent className="space-y-3">
                  {imageMappings.map((mapping) => (
                    <div
                      key={mapping.id}
                      className={cn(
                        "cursor-pointer rounded-[22px] border-2 p-4 transition-all duration-300",
                        selectedMapping === mapping.id
                          ? "border-primary bg-primary/5 shadow-md"
                          : "border-border hover:border-primary/50 hover:bg-muted/50"
                      )}
                      onClick={() => setSelectedMapping(selectedMapping === mapping.id ? null : mapping.id)}
                      onMouseEnter={() => setHoveredMapping(mapping.id)}
                      onMouseLeave={() => setHoveredMapping(null)}
                    >
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <Badge 
                          variant="outline" 
                          className={cn(
                            "text-xs",
                            mapping.color.replace('/30', '/100').replace('border-', 'text-')
                          )}
                        >
                          {mapping.region}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {(mapping.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-sm text-foreground leading-relaxed">
                        {mapping.sentence}
                      </p>
                      {selectedMapping === mapping.id && (
                        <div className="mt-3 flex items-center gap-2 border-t border-border pt-3 text-xs text-primary">
                          <Info className="w-3 h-3" />
                          Region highlighted on image
                        </div>
                      )}
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Legend */}
              <Card className="surface-card">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Color Legend</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {imageMappings.map((mapping) => (
                    <div key={mapping.id} className="flex items-center gap-2">
                      <div className={cn("w-4 h-4 rounded border-2", mapping.color)} />
                      <span className="text-sm text-foreground">{mapping.region}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Navigation */}
              <Card className="surface-card">
                <CardContent className="p-4">
                  <Link to="/knowledge-graph">
                    <Button variant="outline" className="w-full justify-between">
                      <span>Explore Knowledge Graph</span>
                      <ChevronRight className="w-4 h-4" />
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
};

export default ImageMapping;
