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
  ArrowRight,
  UserCheck
} from "lucide-react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
import { usePatientContext } from "@/context/PatientContext";
import FeedbackPanel from "@/components/FeedbackPanel";
import KnowledgeGraph from "@/components/KnowledgeGraph";
import { supabase } from "@/lib/supabaseClient";

type ProcessingStep = 'idle' | 'uploading' | 'analyzing' | 'complete';
type ScanType = 'auto' | 'xray' | 'ct';

interface ExtendedParsedReport extends ParsedReport {
  recommendation?: string;
}

const XRAY_AGENT_STEPS = [
  { id: 'vision', label: 'Vision Agent', description: 'Extracting relevant visual features & embeddings...', icon: Scan, color: 'text-blue-500', progress: 25 },
  { id: 'draft', label: 'Retrieval and Draft Agent', description: 'Fetching similar cases from vector DB and generating draft report...', icon: Database, color: 'text-amber-500', progress: 50 },
  { id: 'kg', label: 'Knowledge Graph Agent', description: 'Extracting clinical entities & relationships from input scan...', icon: Network, color: 'text-purple-500', progress: 75 },
  { id: 'synthesis', label: 'Synthesis Agent', description: 'Composing detailed final narrative report...', icon: Sparkles, color: 'text-emerald-500', progress: 90 }
];

const CT_AGENT_STEPS = [
  { id: 'kg', label: 'Knowledge Graph Agent', description: 'Mapping CT scan to clinical knowledge graph...', icon: Network, color: 'text-purple-500', progress: 20 },
  { id: 'vision', label: 'CT Vision Agent (MedGemma)', description: 'Building CT slice montage and running MedGemma inference...', icon: Brain, color: 'text-blue-500', progress: 55 },
  { id: 'synthesis', label: 'Report Synthesis', description: 'Structuring CT findings into FINDINGS / IMPRESSION format...', icon: Sparkles, color: 'text-emerald-500', progress: 90 }
];

const AGENT_STEPS = XRAY_AGENT_STEPS; // Legacy alias

const getAgentSteps = (scanType: string) =>
  scanType === 'ct' ? CT_AGENT_STEPS : XRAY_AGENT_STEPS;


const UploadXray = () => {
  const {
    uploadedFile,
    previewUrl,
    report,
    knowledgeGraphData,
    setAnalysisResults,
    resetAnalysis
  } = useAnalysis();

  const { selectedPatient, pendingUploadFiles, setPendingUploadFiles, pendingScanType, setPendingScanType } = usePatientContext();

  const [tempFiles, setTempFiles] = useState<File[]>([]);
  const [tempPreviews, setTempPreviews] = useState<string[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [currentStep, setCurrentStep] = useState<ProcessingStep>(report ? 'complete' : 'idle');
  const [activeAgentIndex, setActiveAgentIndex] = useState(0);
  const [progress, setProgress] = useState(report ? 100 : 0);

  // NEW: State to track which scan type the user selected
  const [scanType, setScanType] = useState<ScanType>('auto');

  const displayFile = uploadedFile || tempFiles[0];
  const displayUrl = previewUrl || tempPreviews[0];

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

  const processFiles = useCallback(async (files: FileList | File[]) => {
    if (!selectedPatient) {
      alert("Please select a patient from the Patients dashboard first.");
      return;
    }

    // Cleanup old previews
    tempPreviews.forEach(p => URL.revokeObjectURL(p));

    // Convert to array and handle previews
    const fileArray = Array.from(files);
    const newPreviews = fileArray.map(f => URL.createObjectURL(f));
    setTempFiles(fileArray);
    setTempPreviews(newPreviews);

    setCurrentStep('uploading');
    setProgress(10);
    setActiveAgentIndex(0);

    const formData = new FormData();
    fileArray.forEach(f => formData.append("files", f));
    // NEW: Send the selected scan type to the backend
    formData.append("scan_type", scanType);

    try {
      setCurrentStep('analyzing');
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
      const response = await fetch(`${API_BASE_URL}/api/synthesize-report`, {
        method: "POST",
        body: formData,
        headers: { "ngrok-skip-browser-warning": "true" }
      });

      if (!response.ok) throw new Error("Synthesis failed");
      if (!response.body) throw new Error("No readable stream");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finishedSuccessfully = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          if (!finishedSuccessfully) {
            throw new Error("The backend server disconnected prematurely before finishing the analysis.");
          }
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            const currentSteps = getAgentSteps(scanType);

            const updateStepInfo = (stepId: string) => {
              const idx = currentSteps.findIndex(s => s.id === stepId);
              if (idx !== -1) {
                setActiveAgentIndex(idx);
                setProgress(currentSteps[idx].progress);
              }
            };

            if (data.status === "vision_start") updateStepInfo("vision");
            else if (data.status === "draft_start") updateStepInfo("draft");
            else if (data.status === "kg_start") updateStepInfo("kg");
            else if (data.status === "synthesis_start") updateStepInfo("synthesis");
            else if (data.status === "error") {
              alert(data.message);
              setTempFiles([]);
              setTempPreviews([]);
              setCurrentStep('idle');
              setProgress(0);
              return; // Stop processing
            }
            else if (data.status === "complete") {
              finishedSuccessfully = true;
              const parsedReport = parseReportText(data.final_report);
              // Set analysis results to use the first image for the main report view card
              setAnalysisResults(fileArray[0], newPreviews[0], parsedReport, data.knowledge_graph, data.heatmap);
              setTempFiles([]);
              setTempPreviews([]);
              setProgress(100);
              setCurrentStep('complete');

              // Now save to Supabase
              try {
                // Get Auth User ID since RLS requires user_id
                const { data: { user } } = await supabase.auth.getUser();
                if (!user) throw new Error("Not authenticated");

                const now = new Date().toISOString();
                const fileExt = fileArray[0].name.split('.').pop() || 'png';
                const baseFileName = `${selectedPatient.id}_${now}`.replace(/[:.]/g, '-');
                const origFileName = `${baseFileName}.${fileExt}`;
                const heatFileName = `${baseFileName}_heatmap.png`;

                let original_image_url: string | null = null;
                let heatmap_image_url: string | null = null;

                // 1. Upload ALL original images
                const uploadedUrls: string[] = [];
                for (let i = 0; i < fileArray.length; i++) {
                  const currentFile = fileArray[i];
                  const fileExt = currentFile.name.split('.').pop() || 'png';
                  const origFileName = `${baseFileName}_${i}.${fileExt}`;

                  const { data: origData, error: origErr } = await supabase.storage
                    .from('medical-images')
                    .upload(`${user.id}/${origFileName}`, currentFile);

                  if (!origErr && origData) {
                    const { data: pubOrig } = supabase.storage
                      .from('medical-images')
                      .getPublicUrl(`${user.id}/${origFileName}`);
                    uploadedUrls.push(pubOrig.publicUrl);
                  }
                }

                if (uploadedUrls.length > 0) {
                  original_image_url = uploadedUrls.join(',');
                }

                // 2. Upload heatmap if we have one
                if (data.heatmap) {
                  // data.heatmap is base64 like: "data:image/png;base64,....."
                  const base64Data = data.heatmap.split('base64,')[1];
                  if (base64Data) {
                    const byteCharacters = atob(base64Data);
                    const byteNumbers = new Array(byteCharacters.length);
                    for (let i = 0; i < byteCharacters.length; i++) {
                      byteNumbers[i] = byteCharacters.charCodeAt(i);
                    }
                    const byteArray = new Uint8Array(byteNumbers);
                    const blob = new Blob([byteArray], { type: 'image/png' });

                    const { data: heatData, error: heatErr } = await supabase.storage
                      .from('medical-images')
                      .upload(`${user.id}/${heatFileName}`, blob);

                    if (!heatErr && heatData) {
                      const { data: pubHeat } = supabase.storage
                        .from('medical-images')
                        .getPublicUrl(`${user.id}/${heatFileName}`);
                      heatmap_image_url = pubHeat.publicUrl;
                    }
                  }
                }

                // 3. Insert into medical_scans
                // Default severity to 'moderate' for now, could be derived from labels if needed
                let severity = 'moderate';
                const lowerImpression = parsedReport.impression.toLowerCase();
                const lowerFindings = parsedReport.findings.toLowerCase();
                if (lowerImpression.includes('normal') || lowerFindings.includes('unremarkable')) severity = 'mild';
                if (lowerImpression.includes('severe') || lowerImpression.includes('critical')) severity = 'severe';


                const { error: insertErr } = await supabase.from('medical_scans').insert([{
                  patient_id: selectedPatient.id,
                  user_id: user.id,
                  scan_type: scanType, // 'auto', 'ct', or 'xray'
                  original_image_url,
                  heatmap_image_url,
                  findings: parsedReport.findings,
                  impression: parsedReport.impression,
                  recommendation: parsedReport.recommendation || null,
                  labels: parsedReport.labels,
                  kg_data: data.knowledge_graph || null,
                  severity
                }]);

                if (insertErr) {
                  console.error("Failed to save report to database:", insertErr);
                  // Don't alert the user necessarily to interrupt the flow, just log it.
                }

              } catch (dbError) {
                console.error("Database Save Error:", dbError);
              }
            }
          } catch (e) {
            console.error("Error parsing stream line", e);
          }
        }
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Analysis failed. Ensure backend is running.");
      tempPreviews.forEach(p => URL.revokeObjectURL(p));
      setTempFiles([]);
      setTempPreviews([]);
      setCurrentStep('idle');
      setProgress(0);
    }
  }, [setAnalysisResults, tempPreviews, scanType, selectedPatient]);

  // AUTO-TRIGGER: When coming from Patient Dashboard via Upload Scan button
  useEffect(() => {
    if (pendingUploadFiles && pendingUploadFiles.length > 0) {
      // 1. Set the scan type from the dialog selection BEFORE processing
      setScanType(pendingScanType as ScanType);
      // 2. Clear from context so it doesn't re-fire on re-render
      const files = pendingUploadFiles;
      setPendingUploadFiles(null);
      setPendingScanType('auto');
      // 3. Small delay so the state update settles, then fire
      setTimeout(() => processFiles(files), 50);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only on mount


  const handleReset = useCallback(() => {
    tempPreviews.forEach(p => URL.revokeObjectURL(p));
    resetAnalysis();
    setTempFiles([]);
    setTempPreviews([]);
    setCurrentStep('idle');
    setProgress(0);
    setActiveAgentIndex(0);
  }, [resetAnalysis, tempPreviews]);

  const extendedReport = report as ExtendedParsedReport | null;
  const { feedbackStatus, setFeedbackStatus } = useAnalysis();

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
                <CardContent className="p-4">
                  <Tabs
                    defaultValue="auto"
                    value={scanType}
                    onValueChange={(val) => setScanType(val as ScanType)}
                    className="w-full max-w-sm mx-auto mb-6"
                  >
                    <TabsList className="grid w-full grid-cols-3 bg-slate-800 text-slate-400">
                      <TabsTrigger value="auto" className="data-[state=active]:bg-slate-700 data-[state=active]:text-white">Auto</TabsTrigger>
                      <TabsTrigger value="xray" className="data-[state=active]:bg-slate-700 data-[state=active]:text-white">X-Ray</TabsTrigger>
                      <TabsTrigger value="ct" className="data-[state=active]:bg-slate-700 data-[state=active]:text-white">CT Scan</TabsTrigger>
                    </TabsList>
                  </Tabs>

                  {!displayFile ? (
                    <>
                      {/* NEW: Scan Type Selector */}
                      {/* This section is now replaced by the Tabs component above */}
                      {/* <div className="mb-4">
                        <label className="text-sm font-semibold text-muted-foreground block mb-2 text-center uppercase tracking-wider">
                          Select Scan Type
                        </label>
                        <div className="flex justify-center gap-2">
                          <Button
                            variant={scanType === 'auto' ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setScanType('auto')}
                          >
                            <Brain className="w-4 h-4 mr-2" /> Auto-Detect
                          </Button>
                          <Button
                            variant={scanType === 'xray' ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setScanType('xray')}
                          >
                            <Scan className="w-4 h-4 mr-2" /> Chest X-Ray
                          </Button>
                          <Button
                            variant={scanType === 'ct' ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setScanType('ct')}
                          >
                            <Database className="w-4 h-4 mr-2" /> CT Scan
                          </Button>
                        </div>
                      </div>

                      {/* Drag & Drop Area */}
                      <div
                        className={cn(
                          "relative p-12 border-2 border-dashed rounded-lg transition-all",
                          !selectedPatient ? "opacity-50 cursor-not-allowed border-border" : "cursor-pointer",
                          dragActive && selectedPatient ? "border-primary bg-primary/5" : (!selectedPatient ? "" : "border-border hover:border-primary/40")
                        )}
                        onDragEnter={selectedPatient ? handleDrag : undefined}
                        onDragLeave={selectedPatient ? handleDrag : undefined}
                        onDragOver={selectedPatient ? handleDrag : undefined}
                        onDrop={selectedPatient ? ((e) => {
                          e.preventDefault();
                          if (e.dataTransfer.files?.length > 0) processFiles(e.dataTransfer.files);
                        }) : undefined}
                        onClick={selectedPatient ? (() => document.getElementById('file-input')?.click()) : undefined}
                      >
                        <input
                          id="file-input"
                          type="file"
                          className="hidden"
                          accept="image/*"
                          multiple
                          disabled={!selectedPatient}
                          onChange={(e) => e.target.files?.length && processFiles(e.target.files)}
                        />
                        <div className="text-center">
                          <Upload className="w-12 h-12 mx-auto mb-4 text-primary" />
                          <h3 className="text-lg font-semibold">Upload Image</h3>
                          <p className="text-sm text-muted-foreground mt-1">
                            {!selectedPatient ? "Select a patient first to unlock" : "Drag & drop or click to browse"}
                          </p>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div>
                      <div className="relative aspect-square rounded-lg overflow-hidden bg-black mb-4">
                        <img src={displayUrl!} alt="Scan" className="w-full h-full object-contain" />

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
                        <div className="flex gap-2">
                          <span className="px-2 py-1 bg-muted rounded text-xs uppercase font-bold flex items-center">
                            Mode: {scanType}
                          </span>
                          <Button variant="ghost" size="sm" onClick={handleReset} disabled={currentStep === 'analyzing'}>
                            <RefreshCw className={cn("w-3 h-3 mr-1", currentStep === 'analyzing' && "animate-spin")} />
                            {currentStep === 'analyzing' ? 'Busy...' : 'Reset'}
                          </Button>
                        </div>
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
                          (getAgentSteps(scanType)[activeAgentIndex]?.label ?? 'Processing...')}
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
                  </div>

                  {/* HITL Feedback */}
                  {feedbackStatus === 'none' && (
                    <Button
                      variant="outline"
                      className="w-full gap-2 border-amber-500/30 text-amber-600 hover:bg-amber-500/10 hover:border-amber-500/50"
                      onClick={() => setFeedbackStatus('reviewing')}
                    >
                      <UserCheck className="w-4 h-4" />
                      Edit &amp; Review Report
                    </Button>
                  )}
                  {(feedbackStatus === 'reviewing' || feedbackStatus === 'draft' || feedbackStatus === 'approved') && (
                    <FeedbackPanel onReAnalyze={handleReset} />
                  )}
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
                        {getAgentSteps(scanType).map((step, index) => {
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

          {/* KNOWLEDGE GRAPH FULL WIDTH SECTION */}
          {extendedReport && knowledgeGraphData && (
            <div className="mt-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
              <KnowledgeGraph data={knowledgeGraphData} />
            </div>
          )}
        </div>
      </section>
    </Layout>
  );
};

export default UploadXray;