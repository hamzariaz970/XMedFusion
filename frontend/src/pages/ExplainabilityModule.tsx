import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FileSearch, Info, Activity, Sparkles } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { cn } from "@/lib/utils";

// 1. Import the Context to get the stored data
import { useAnalysis } from "@/context/AnalysisContext";

// Use the same env var as UploadXray so Vercel deployments work correctly
const BACKEND_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const ExplainabilityModule = () => {
  const navigate = useNavigate();

  // 2. Get Data directly from Global Context
  // We don't need to re-fetch because UploadXray already saved it here.
  const { previewUrl, heatmapData, report } = useAnalysis();

  // Parse the report to extract findings and impression for context
  const extendedReport = report as any; // Cast temporarily if interface isn't exported here
  const findings = extendedReport?.findings || "No findings recorded.";
  const impression = extendedReport?.impression || "No impression recorded.";

  const handleBackToUpload = () => navigate("/upload");

  // 3. Query the Python Backend for the Explanation
  const { data: aiExplanation, isLoading: isAiLoading } = useQuery({
    queryKey: ["geminiAnalysis", findings, impression],
    queryFn: async () => {
      if (!findings || !impression) return null;

      const response = await fetch(`${BACKEND_URL}/api/explain`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          findings,
          impression
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate explanation');
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }

      return data.explanation;
    },
    enabled: !!previewUrl,
  });

  return (
    <Layout>
      <div className="figma-page-shell min-h-[80vh]">
        {/* Header Section */}
        <div className="figma-workspace-hero mb-8 grid w-full gap-6 text-left lg:grid-cols-[1fr_380px] lg:items-center">
          <div>
            <Badge variant="outline" className="eyebrow mb-4">
              <Sparkles className="w-4 h-4" /> AI Interpretability
            </Badge>
            <h1 className="mb-3 text-3xl font-extrabold tracking-tight md:text-5xl">
              Explainability <span className="text-primary">Module</span>
            </h1>
            <p className="max-w-2xl text-muted-foreground">
              Visualize the evidence behind generated reports with original imaging, model highlights, and an AI narrative.
            </p>
          </div>
          <RadiologyImageCard
            src={previewUrl || radiologyImages.neuroReview}
            alt="Radiology explainability review"
            label="Visual evidence"
            caption={previewUrl ? "Current scan loaded" : "Heatmaps and narrative review"}
            className="min-h-[240px]"
          />
        </div>

        {!previewUrl ? (
          <Card className="surface-card mx-auto max-w-2xl border-dashed">
            <CardContent className="flex flex-col items-center p-10">
              <FileSearch className="w-12 h-12 mb-4 text-muted-foreground opacity-50" />
              <p className="mb-6 text-center">
                No image data found. Please upload an X-ray to begin analysis.
              </p>
              <Button onClick={handleBackToUpload}>Return to Upload</Button>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-8">
            <div className="grid w-full grid-cols-1 gap-8 lg:grid-cols-2">

              {/* CARD 1: Original Image */}
              <Card className="surface-card overflow-hidden">
                <CardHeader className="border-b border-border/50 bg-secondary/40">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Activity className="w-4 h-4 text-primary" /> Original Scan
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex aspect-square items-center justify-center bg-clinical-ink p-4">
                  <img
                    src={previewUrl}
                    alt="Original X-ray"
                    className="max-h-full rounded-[18px] object-contain"
                  />
                </CardContent>
              </Card>

              {/* CARD 2: Insights Heatmap / Bounding Boxes */}
              <Card className="surface-card overflow-hidden">
                <CardHeader className="border-b border-border/50 bg-secondary/40">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Info className="w-4 h-4 text-primary" /> Visual Insights
                  </CardTitle>
                </CardHeader>
                <CardContent className="relative flex aspect-square items-center justify-center bg-clinical-ink p-4">
                  {/* DIRECTLY USE heatmapData FROM CONTEXT */}
                  {heatmapData ? (
                    <img
                      src={heatmapData}
                      alt="Heatmap"
                      className="max-h-full rounded-[18px] object-contain opacity-100 transition-opacity duration-700"
                    />
                  ) : (
                    <div className="text-white text-center text-sm opacity-70 px-4">
                      <p className="font-medium mb-2">No visual highlights available.</p>
                      <p className="text-xs opacity-50">
                        Bounding boxes are generated by the Knowledge Graph Agent. CT scans do not currently have KG Agent support.
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* CARD 3: AI Narrative Result Section */}
              <Card className="surface-card lg:col-span-2 border-primary/20 shadow-lg overflow-hidden">
                <div className="h-1 bg-gradient-to-r from-primary via-blue-400 to-primary animate-pulse" />
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-primary" />
                    Automated Diagnostic Narrative
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {isAiLoading ? (
                    <div className="space-y-3">
                      <Skeleton className="h-4 w-full" />
                      <Skeleton className="h-4 w-[90%]" />
                      <Skeleton className="h-4 w-[95%]" />
                      <Skeleton className="h-20 w-full" />
                    </div>
                  ) : aiExplanation ? (
                    <ScrollArea className="h-[350px] w-full pr-4">
                      <div className="space-y-5">
                        {/* Split on bold markdown "**Step N:**" or "Step N:" patterns */}
                        {aiExplanation
                          .split(/\n(?=\*{0,2}Step\s+\d+)/i)
                          .filter((block: string) => block.trim())
                          .map((block: string, idx: number) => {
                            // Strip leading/trailing ** from heading line
                            const lines = block.trim().split('\n');
                            const rawHeading = lines[0].replace(/^\*{1,2}|\*{1,2}$/g, '').trim();
                            const body = lines.slice(1).join('\n').replace(/\*{1,2}/g, '').trim();
                            return (
                              <div key={idx} className="rounded-[22px] border border-primary/10 bg-primary/5 p-4">
                                <h4 className="text-sm font-bold text-primary mb-2 flex items-center gap-2">
                                  <Activity className="w-4 h-4 shrink-0" />
                                  {rawHeading}
                                </h4>
                                {body && (
                                  <p className="text-sm leading-relaxed text-foreground/80 whitespace-pre-wrap">
                                    {body}
                                  </p>
                                )}
                              </div>
                            );
                          })
                        }
                      </div>
                    </ScrollArea>
                  ) : (
                    <p className="text-sm text-muted-foreground italic">Narrative not available.</p>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default ExplainabilityModule;
