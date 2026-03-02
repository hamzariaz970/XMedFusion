import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FileSearch, Info, Activity, Sparkles } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { cn } from "@/lib/utils";

// 1. Import the Context to get the stored data
import { useAnalysis } from "@/context/AnalysisContext";

// Localhost backend URL
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

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
      <div className="container mx-auto px-4 py-12 lg:py-20 min-h-[80vh]">
        {/* Header Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4">
            <Sparkles className="w-4 h-4" /> AI Interpretability
          </div>
          <h1 className="text-3xl md:text-4xl font-bold mb-2">
            Explainability Module
          </h1>
          <p className="text-muted-foreground">
            Visualizing the "Why" behind AI-generated medical reports.
          </p>
        </div>

        {!previewUrl ? (
          <Card className="max-w-md mx-auto border-dashed">
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
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">

              {/* CARD 1: Original Image */}
              <Card className="overflow-hidden border-2">
                <CardHeader className="border-b bg-muted/30">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Activity className="w-4 h-4 text-primary" /> Original Scan
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4 bg-black aspect-square flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="Original X-ray"
                    className="max-h-full object-contain rounded-sm"
                  />
                </CardContent>
              </Card>

              {/* CARD 2: Insights Heatmap */}
              <Card className="overflow-hidden border-2">
                <CardHeader className="border-b bg-muted/30">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Info className="w-4 h-4 text-primary" /> Visual Insights
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4 bg-black aspect-square flex items-center justify-center relative">
                  {/* DIRECTLY USE heatmapData FROM CONTEXT */}
                  {heatmapData ? (
                    <img
                      src={heatmapData}
                      alt="Heatmap"
                      className="max-h-full object-contain rounded-sm opacity-100 transition-opacity duration-700"
                    />
                  ) : (
                    <div className="text-white text-center text-sm opacity-70">
                      Heatmap data not found in context.
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* CARD 3: AI Narrative Result Section */}
              <Card className="lg:col-span-2 border-primary/20 shadow-lg overflow-hidden">
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
                              <div key={idx} className="rounded-lg border border-primary/10 bg-muted/20 p-4">
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