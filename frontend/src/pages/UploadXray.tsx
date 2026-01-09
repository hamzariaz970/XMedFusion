import { useState, useCallback, useEffect } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { 
  Upload, 
  Loader2, 
  Brain,
  FileText,
  Sparkles,
  RefreshCw,
  Tag,
  Stethoscope,
  Scan,        
  Database,    
  Network,     
  CheckCircle2,
  Eye,
  Download,
  FileDown,
  ArrowRight
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { Link } from "react-router-dom";
import jsPDF from "jspdf"; 

// Import Global Context
import { useAnalysis, ParsedReport } from "@/context/AnalysisContext";

type ProcessingStep = 'idle' | 'uploading' | 'analyzing' | 'complete';

interface ExtendedParsedReport extends ParsedReport {
  recommendation?: string;
}

const AGENT_STEPS = [
  { id: 'vision', label: 'Vision Agent', description: 'Extracting relevant visual features & embeddings...', icon: Scan, color: 'text-blue-500', progress: 25 },
  { id: 'draft', label: 'Retrieval and Draft Agent', description: 'Fetching similar cases from vector DB and generating draft report...', icon: Database, color: 'text-amber-500', progress: 50 },
  { id: 'kg', label: 'Knowledge Graph Agent', description: 'Extracting clinical entities & relationships from input scan...', icon: Network, color: 'text-purple-500', progress: 75 },
  { id: 'synthesis', label: 'Synthesis Agent', description: 'Composing detailed final narrative report...', icon: Sparkles, color: 'text-emerald-500', progress: 90 }
];

const UploadXray = () => {
  const { 
    uploadedFile, 
    previewUrl, 
    report, 
    setAnalysisResults, 
    resetAnalysis 
  } = useAnalysis();

  const [tempFile, setTempFile] = useState<File | null>(null);
  const [tempPreview, setTempPreview] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [currentStep, setCurrentStep] = useState<ProcessingStep>(report ? 'complete' : 'idle');
  const [activeAgentIndex, setActiveAgentIndex] = useState(0);
  const [progress, setProgress] = useState(report ? 100 : 0);

  const displayFile = uploadedFile || tempFile;
  const displayUrl = previewUrl || tempPreview;

  const parseReportText = (text: string): ExtendedParsedReport => {
    const findingsMatch = text.match(/FINDINGS:([\s\S]*?)(?=IMPRESSIONS?:|$)/i);
    const impressionMatch = text.match(/IMPRESSIONS?:([\s\S]*?)(?=RECOMMENDATIONS?:|LABELS:|$)/i);
    const recommendationMatch = text.match(/RECOMMENDATIONS?:([\s\S]*?)(?=LABELS:|$)/i);
    const labelsMatch = text.match(/LABELS:([\s\S]*?)$/i);

    return {
      findings: findingsMatch?.[1]?.trim() || "Findings content not found.",
      impression: impressionMatch?.[1]?.trim() || "Impression content not found.",
      recommendation: recommendationMatch?.[1]?.trim(),
      labels: labelsMatch?.[1]
        ? labelsMatch[1].split(',').map(l => l.trim()).filter(l => l !== "")
        : []
    };
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  }, []);

  const processFile = useCallback(async (file: File) => {
    if (tempPreview) URL.revokeObjectURL(tempPreview);
    const objectUrl = URL.createObjectURL(file);
    setTempFile(file);
    setTempPreview(objectUrl);
    setCurrentStep('uploading');
    setProgress(10);
    setActiveAgentIndex(0);

    const formData = new FormData();
    formData.append("file", file);

    try {
      setCurrentStep('analyzing');
      const response = await fetch("http://localhost:8000/api/synthesize-report", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Synthesis failed");
      if (!response.body) throw new Error("No readable stream");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; 

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.status === "vision_start") { setActiveAgentIndex(0); setProgress(AGENT_STEPS[0].progress); } 
            else if (data.status === "draft_start") { setActiveAgentIndex(1); setProgress(AGENT_STEPS[1].progress); } 
            else if (data.status === "kg_start") { setActiveAgentIndex(2); setProgress(AGENT_STEPS[2].progress); } 
            else if (data.status === "synthesis_start") { setActiveAgentIndex(3); setProgress(AGENT_STEPS[3].progress); } 
            else if (data.status === "complete") {
              const parsedReport = parseReportText(data.final_report);
              setAnalysisResults(file, objectUrl, parsedReport, data.knowledge_graph, data.heatmap);
              setTempFile(null);
              setTempPreview(null);
              setProgress(100);
              setCurrentStep('complete');
            }
          } catch (e) { console.error("Error parsing stream line", e); }
        }
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Analysis failed. Ensure backend is running.");
      URL.revokeObjectURL(objectUrl);
      setTempFile(null);
      setTempPreview(null);
      setCurrentStep('idle');
      setProgress(0);
    }
  }, [setAnalysisResults, tempPreview]);

  const handleReset = useCallback(() => {
    if (tempPreview) URL.revokeObjectURL(tempPreview);
    resetAnalysis();     
    setTempFile(null);   
    setTempPreview(null);
    setCurrentStep('idle');
    setProgress(0);
    setActiveAgentIndex(0);
  }, [resetAnalysis, tempPreview]);

  const extendedReport = report as ExtendedParsedReport | null;

  // ------------------------------------------------------------------
  // DOWNLOAD HANDLERS (FIXED PDF WRAPPING)
  // ------------------------------------------------------------------

  const downloadTXT = () => {
    if (!extendedReport) return;
    const content = `X-MEDFUSION AI RADIOLOGY REPORT\nDate: ${new Date().toLocaleString()}\nFile: ${displayFile?.name || 'Unknown'}\n\nFINDINGS:\n${extendedReport.findings}\n\nIMPRESSION:\n${extendedReport.impression}\n\nRECOMMENDATION:\n${extendedReport.recommendation || "None"}\n\nLABELS:\n${extendedReport.labels.join(", ")}\n\n---------------------------------------------------\nDISCLAIMER: Research use only. Not a medical diagnosis.`;
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Report_${displayFile?.name.split('.')[0] || 'Xray'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadPDF = async () => {
    if (!extendedReport) return;
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    
    // Config
    const margin = 20;
    // Reduce width slightly more to ensure safe right margin
    const maxLineWidth = pageWidth - (margin * 2.2); 
    const lineHeight = 7;

    let yPos = 20;

    // 1. Header
    doc.setFontSize(22);
    doc.setFont("helvetica", "bold");
    doc.text("X-MedFusion AI Report", margin, yPos);
    yPos += 10;
    
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(100);
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, yPos);
    yPos += 5;
    doc.text(`File: ${displayFile?.name || 'Unknown'}`, margin, yPos);
    yPos += 15;

    // 2. Add Image (With aspect ratio protection)
    if (displayUrl) {
      try {
        const img = new Image();
        img.src = displayUrl;
        await new Promise((resolve) => { img.onload = resolve; });
        
        const imgProps = doc.getImageProperties(img);
        const pdfImgWidth = 60; // mm
        const pdfImgHeight = (imgProps.height * pdfImgWidth) / imgProps.width;
        
        // Check if image fits on page, else add page
        if (yPos + pdfImgHeight > pageHeight - margin) {
            doc.addPage();
            yPos = margin;
        }

        doc.addImage(img, 'PNG', margin, yPos, pdfImgWidth, pdfImgHeight);
        yPos += pdfImgHeight + 10;
      } catch (e) {
        console.error("Could not add image to PDF", e);
      }
    }

    doc.setTextColor(0); 

    // Helper: Add Section with Auto-Wrapping and Auto-Paging
    const addSection = (title: string, content: string) => {
        // Check if title fits
        if (yPos > pageHeight - margin) { doc.addPage(); yPos = margin; }
        
        doc.setFontSize(12);
        doc.setFont("helvetica", "bold");
        doc.text(title, margin, yPos);
        yPos += 6;
        
        doc.setFontSize(11);
        doc.setFont("helvetica", "normal");
        
        // CRITICAL: Split text into lines that fit within maxLineWidth
        const lines = doc.splitTextToSize(content, maxLineWidth);
        
        // Check if content fits, if not, handle paging
        // (Simple approach: if huge block, just dump it. For better PDF, iterate lines)
        if (yPos + (lines.length * 6) > pageHeight - margin) {
             doc.addPage(); 
             yPos = margin; 
        }

        doc.text(lines, margin, yPos);
        yPos += (lines.length * 6) + 6; // Spacing after section
    };

    // 3. Report Sections
    addSection("FINDINGS", extendedReport.findings);
    addSection("IMPRESSION", extendedReport.impression);
    
    if (extendedReport.recommendation) {
        addSection("RECOMMENDATION", extendedReport.recommendation);
    }

    // 4. Labels (FIXED: Now wrapped!)
    if (yPos > pageHeight - margin) { doc.addPage(); yPos = margin; }
    
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.text("DETECTED LABELS", margin, yPos);
    yPos += 6;
    
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(50);
    
    // FIX: Split label string into wrapped lines
    const labelString = extendedReport.labels.join(", ");
    const labelLines = doc.splitTextToSize(labelString, maxLineWidth);
    
    doc.text(labelLines, margin, yPos);
    yPos += (labelLines.length * 5) + 15;

    // 5. Disclaimer Footer
    const disclaimer = "DISCLAIMER: This report is generated by an automated AI system (X-MedFusion). It is intended for research and educational purposes only and does not constitute a valid medical diagnosis.";
    const discLines = doc.splitTextToSize(disclaimer, maxLineWidth);
    
    // Always put disclaimer at bottom of current page, or new page if full
    if (yPos > pageHeight - 30) { doc.addPage(); }
    
    const footerY = doc.internal.pageSize.getHeight() - 20;
    doc.setTextColor(150);
    doc.setFontSize(8);
    doc.setFont("helvetica", "italic");
    doc.text(discLines, margin, footerY);

    doc.save(`XMedFusion_Report_${displayFile?.name.split('.')[0] || 'Scan'}.pdf`);
  };

  return (
    <Layout>
      <section className="py-12 lg:py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              AI Report <span className="text-primary">Synthesis</span>
            </h1>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
            {/* LEFT COLUMN: UPLOAD */}
            <div className="space-y-6">
              <Card className="overflow-hidden border-2">
                <CardContent className="p-0">
                  {!displayFile ? (
                    <div
                      className={cn(
                        "relative p-12 border-2 border-dashed rounded-lg m-4 transition-all cursor-pointer",
                        dragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/40"
                      )}
                      onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag}
                      onDrop={(e) => {
                        e.preventDefault();
                        if (e.dataTransfer.files?.[0]) processFile(e.dataTransfer.files[0]);
                      }}
                      onClick={() => document.getElementById('file-input')?.click()}
                    >
                      <input id="file-input" type="file" className="hidden" accept="image/*" onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])} />
                      <div className="text-center">
                        <Upload className="w-12 h-12 mx-auto mb-4 text-primary" />
                        <h3 className="text-lg font-semibold">Upload X-ray Image</h3>
                      </div>
                    </div>
                  ) : (
                    <div className="p-4">
                      <div className="relative aspect-square rounded-lg overflow-hidden bg-black mb-4">
                        <img src={displayUrl!} alt="X-ray" className="w-full h-full object-contain" />
                        
                        {currentStep === 'analyzing' && (
                          <div className="absolute inset-0 bg-background/60 backdrop-blur-sm flex flex-col items-center justify-center text-center p-4">
                            <Brain className="w-12 h-12 text-primary animate-pulse mb-4" />
                            <p className="text-lg font-bold">X-MedFusion Processing</p>
                            <p className="text-sm text-muted-foreground mt-1">Multi-Agent System Active</p>
                          </div>
                        )}
                      </div>
                      <div className="flex justify-between items-center text-sm">
                        <span className="truncate max-w-[200px]">{displayFile.name}</span>
                        <Button variant="ghost" size="sm" onClick={handleReset} disabled={currentStep === 'analyzing'}>
                          <RefreshCw className={cn("w-3 h-3 mr-1", currentStep === 'analyzing' && "animate-spin")} /> 
                          {currentStep === 'analyzing' ? 'Busy...' : 'Reset'}
                        </Button>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {currentStep !== 'idle' && (
                <div className="space-y-2 animate-in fade-in slide-in-from-bottom-2">
                  <div className="flex justify-between text-xs font-bold uppercase tracking-wider text-muted-foreground">
                    <span>
                      {currentStep === 'complete' ? 'Analysis Complete' : 
                       currentStep === 'uploading' ? 'Uploading...' : 
                       AGENT_STEPS[activeAgentIndex].label}
                    </span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              )}
            </div>

            {/* RIGHT COLUMN: REPORT */}
            <div className="space-y-6">
              {extendedReport ? (
                <>
                  <Card className="border-primary/20 shadow-xl animate-in fade-in zoom-in-95 duration-500">
                    <CardHeader className="bg-muted/30 border-b flex flex-row items-center justify-between">
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <FileText className="w-5 h-5 text-primary" />
                        Synthesized Report
                      </CardTitle>
                      
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="outline" size="sm" className="h-8 gap-2">
                            <Download className="w-4 h-4" />
                            Save
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={downloadTXT} className="cursor-pointer">
                            <FileText className="w-4 h-4 mr-2" /> Save as .TXT
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={downloadPDF} className="cursor-pointer">
                            <FileDown className="w-4 h-4 mr-2" /> Save as .PDF
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>

                    </CardHeader>
                    <CardContent className="p-6 space-y-6">
                      
                      <section>
                        <h4 className="text-xs font-black uppercase text-muted-foreground mb-2 flex items-center gap-2">
                          <Brain className="w-3 h-3" /> Findings
                        </h4>
                        <div className="text-sm leading-relaxed bg-muted/20 p-4 rounded-md border whitespace-pre-line">
                          {extendedReport.findings}
                        </div>
                      </section>

                      <section>
                        <h4 className="text-xs font-black uppercase text-muted-foreground mb-2 flex items-center gap-2">
                          <Sparkles className="w-3 h-3" /> Impression
                        </h4>
                        <div className="p-4 bg-primary/5 border border-primary/10 rounded-md italic text-sm font-medium leading-relaxed">
                          {extendedReport.impression}
                        </div>
                      </section>

                      {extendedReport.recommendation && (
                        <section>
                          <h4 className="text-xs font-black uppercase text-muted-foreground mb-2 flex items-center gap-2">
                            <Stethoscope className="w-3 h-3" /> Recommendation
                          </h4>
                          <div className="p-3 bg-amber-500/10 border border-amber-500/20 text-amber-700 dark:text-amber-400 rounded-md text-sm font-medium">
                            {extendedReport.recommendation}
                          </div>
                        </section>
                      )}

                      <section>
                        <h4 className="text-xs font-black uppercase text-muted-foreground mb-3 flex items-center gap-2">
                          <Tag className="w-3 h-3" /> Labels
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {extendedReport.labels.map((label, i) => (
                            <span key={i} className="px-3 py-1 bg-primary text-primary-foreground text-[10px] font-bold rounded-full uppercase">
                              {label}
                            </span>
                          ))}
                        </div>
                      </section>

                    </CardContent>
                  </Card>

                  <div className="flex gap-4">
                    <Link to="/explainability" className="flex-1">
                      <Button variant="outline" className="w-full">
                        Explainability
                        <Eye className="ml-2 w-4 h-4" />
                      </Button>
                    </Link>
                    <Link to="/knowledge-graph" className="flex-1">
                      <Button variant="default" className="w-full">
                        Explore Knowledge Graph
                        <ArrowRight className="ml-2 w-4 h-4" />
                      </Button>
                    </Link>
                  </div>
                </>
              ) : (
                <Card className={cn(
                  "h-full min-h-[500px] flex flex-col transition-all duration-300",
                  currentStep === 'analyzing' ? "border-primary/50 shadow-lg bg-primary/5" : "border-dashed border-2"
                )}>
                  {currentStep === 'analyzing' ? (
                    <CardContent className="flex flex-col justify-center h-full p-8 space-y-8">
                      <div className="text-center mb-4">
                        <h3 className="text-xl font-bold animate-pulse">Orchestrating Agents</h3>
                        <p className="text-muted-foreground text-sm">Processing pipeline in real-time...</p>
                      </div>

                      <div className="space-y-6">
                        {AGENT_STEPS.map((step, index) => {
                          const isActive = index === activeAgentIndex;
                          const isCompleted = index < activeAgentIndex;
                          return (
                            <div key={step.id} className={cn("flex items-center gap-4 p-3 rounded-lg transition-all duration-500", isActive ? "bg-background shadow-md scale-105 border border-primary/20" : "opacity-50")}>
                              <div className={cn("w-10 h-10 rounded-full flex items-center justify-center transition-colors", isActive ? "bg-primary text-primary-foreground" : isCompleted ? "bg-green-500 text-white" : "bg-muted text-muted-foreground")}>
                                {isCompleted ? <CheckCircle2 className="w-5 h-5" /> : isActive ? <Loader2 className="w-5 h-5 animate-spin" /> : <step.icon className="w-5 h-5" />}
                              </div>
                              <div className="flex-1">
                                <div className="flex justify-between items-center"><h4 className={cn("font-bold text-sm", isActive && "text-primary")}>{step.label}</h4></div>
                                <p className="text-xs text-muted-foreground">{step.description}</p>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </CardContent>
                  ) : (
                    <div className="text-center p-12 text-muted-foreground m-auto">
                      <FileText className="w-12 h-12 mx-auto mb-4 opacity-20" />
                      <p className="text-sm font-medium">Report will appear here after analysis.</p>
                    </div>
                  )}
                </Card>
              )}
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
};

export default UploadXray;