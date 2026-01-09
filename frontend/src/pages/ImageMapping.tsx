import { useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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

  return (
    <Layout>
      <section className="py-12 lg:py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Image-Report <span className="text-primary">Mapping</span>
            </h1>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Visualize how report findings are mapped to specific regions of the X-ray image for evidence-based transparency.
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
            {/* Image Display */}
            <div className="lg:col-span-2 space-y-4">
              <Card className="overflow-hidden">
                <CardHeader className="border-b border-border">
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
                  <div className="relative aspect-square bg-foreground/95">
                    {/* Placeholder X-ray image */}
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

                    {/* Region Overlays */}
                    {showOverlays && imageMappings.map((mapping) => (
                      <div
                        key={mapping.id}
                        className={cn(
                          "absolute border-2 rounded-lg cursor-pointer transition-all duration-300",
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
              <Card>
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
                        "p-4 rounded-lg border-2 cursor-pointer transition-all duration-200",
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
                        <div className="mt-3 pt-3 border-t border-border flex items-center gap-2 text-xs text-primary">
                          <Info className="w-3 h-3" />
                          Region highlighted on image
                        </div>
                      )}
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Legend */}
              <Card>
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
              <Card>
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
