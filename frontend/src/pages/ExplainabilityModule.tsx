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

import { useAnalysis } from "@/context/AnalysisContext";
import { getApiBase } from "@/lib/apiConfig";

const ExplainabilityModule = () => {
  const navigate = useNavigate();

  // 2. Get Data directly from Global Context
  // We don't need to re-fetch because UploadXray already saved it here.
  const { previewUrl, referenceImageUrl, heatmapData, report, detectedModality, explainabilityData } = useAnalysis();
  const displayReferenceImage = referenceImageUrl || previewUrl;
  const modalityLabel = detectedModality === "ct" ? "CT scan" : "X-ray";
  const inputPanelLabel = detectedModality === "ct" ? "Model Input Montage" : "Original Scan";
  const inputPanelHint = detectedModality === "ct"
    ? "The exact CT montage used by the CT vision stack"
    : "The original study used for analysis";

  // Parse the report to extract findings and impression for context
  const extendedReport = report as any; // Cast temporarily if interface isn't exported here
  const findings = extendedReport?.findings || "No findings recorded.";
  const impression = extendedReport?.impression || "No impression recorded.";
  const precomputedNarrative = explainabilityData?.automated_narrative || null;

  const handleBackToUpload = () => navigate("/upload");

  // 3. Query the Python Backend for the Explanation
  const { data: aiExplanation, isLoading: isAiLoading } = useQuery({
    queryKey: ["geminiAnalysis", findings, impression, detectedModality],
    queryFn: async () => {
      if (!findings || !impression) return null;

      const API_BASE_URL = await getApiBase();
      const response = await fetch(`${API_BASE_URL}/api/explain`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
          findings,
          impression,
          modality: detectedModality || "xray",
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
    enabled: !!displayReferenceImage && !precomputedNarrative,
    initialData: precomputedNarrative,
  });

  return (
    <Layout>
      <div className="figma-page-shell min-h-[80vh]">
        {/* Header Section */}
        <div className="figma-workspace-hero mb-8 flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Badge variant="outline" className="eyebrow mb-4">
              <Sparkles className="w-4 h-4" /> AI Interpretability
            </Badge>
            <h1 className="mb-3 text-3xl font-extrabold tracking-tight md:text-5xl">
              Explainability <span className="text-primary">Module</span>
            </h1>
            <p className="max-w-2xl text-muted-foreground">
              Visualize the evidence behind generated reports with categorical anatomical highlights and automated diagnostic reasoning.
            </p>
          </div>
          {displayReferenceImage && (
            <div className="w-full lg:w-[380px] p-4 rounded-[28px] border border-primary/20 bg-primary/5">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Evidence Context</p>
              <p className="font-bold text-foreground truncate">{displayReferenceImage.split('/').pop()}</p>
              <p className="text-xs text-muted-foreground mt-1">{inputPanelHint}</p>
            </div>
          )}
        </div>

        {!displayReferenceImage ? (
          <Card className="surface-card mx-auto max-w-2xl border-dashed">
            <CardContent className="flex flex-col items-center p-10">
              <FileSearch className="w-12 h-12 mb-4 text-muted-foreground opacity-50" />
              <p className="mb-6 text-center">
                No image data found. Please upload a scan to begin analysis.
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
                    <Activity className="w-4 h-4 text-primary" /> {inputPanelLabel}
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex aspect-square items-center justify-center bg-clinical-ink p-4">
                  <img
                    src={displayReferenceImage}
                    alt={`${inputPanelLabel} for ${modalityLabel}`}
                    className="h-full max-h-full w-full rounded-[18px] object-contain"
                  />
                </CardContent>
              </Card>

              {/* CARD 2: AI-Annotated Scan */}
              <Card className="surface-card overflow-hidden">
                <CardHeader className="border-b border-border/50 bg-secondary/40">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Info className="w-4 h-4 text-primary" /> AI-Annotated Scan
                  </CardTitle>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Anatomical regions highlighted by categorical findings.{" "}
                    <span className="font-semibold text-primary">Border thickness</span> = Confidence level.
                  </p>
                </CardHeader>
                <CardContent className="relative flex aspect-square items-center justify-center bg-clinical-ink p-4">
                  {/* DIRECTLY USE heatmapData FROM CONTEXT */}
                  {heatmapData ? (
                    <img
                      src={heatmapData}
                      alt="AI-annotated scan with disease bounding boxes"
                      className="max-h-full rounded-[18px] object-contain opacity-100 transition-opacity duration-700"
                    />
                  ) : (
                    <div className="text-white text-center text-sm opacity-70 px-4">
                      <p className="font-medium mb-2">
                        {detectedModality === "ct" ? "No slice-level highlights generated." : "No visual highlights generated."}
                      </p>
                      <p className="text-xs opacity-50">
                        {detectedModality === "ct"
                          ? "The CT report did not return usable slice references, or the study appeared normal."
                          : "The scan may be completely normal, or specific spatial regions could not be mapped."}
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
                  ) : explainabilityData?.reasoning_steps?.length ? (
                    <div className="space-y-3">
                      {explainabilityData.reasoning_steps.map((step: string, idx: number) => (
                        <div key={idx} className="rounded-[22px] border border-primary/10 bg-primary/5 p-4">
                          <p className="text-sm leading-relaxed text-foreground/80">{step}</p>
                        </div>
                      ))}
                    </div>
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
