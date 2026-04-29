import { useState, useEffect, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";
import {
  ChevronLeft, ChevronRight, Save, Send, ArrowLeft,
  FileText, Eye, AlertCircle, CheckCircle, Info, XCircle
} from "lucide-react";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabaseClient";
import { toast } from "sonner";
import { useAuth } from "@/context/AuthContext";

interface HILScan {
  id: string;
  task_id: string;
  image_url: string;
  scan_order: number;
  status: string;
}

interface HILTask {
  id: string;
  title: string;
  instructions: string;
  total_scans: number;
  completed_scans: number;
  status: string;
}

interface ReportDraft {
  indication: string;
  comparison: string;
  findings: string;
  impression: string;
}

const HILLabelingPage = () => {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();

  const [task, setTask] = useState<HILTask | null>(null);
  const [scans, setScans] = useState<HILScan[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [drafts, setDrafts] = useState<Record<string, ReportDraft>>({});
  const [submittedScans, setSubmittedScans] = useState<Set<string>>(new Set());
  const [rejectedFeedback, setRejectedFeedback] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [showInstructions, setShowInstructions] = useState(true);
  const autoSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchTask = useCallback(async () => {
    if (!taskId) return;
    const { data } = await supabase
      .from("hil_tasks")
      .select("*")
      .eq("id", taskId)
      .single();
    if (data) setTask(data);
  }, [taskId]);

  const fetchScans = useCallback(async () => {
    if (!taskId) return;
    const { data } = await supabase
      .from("hil_scans")
      .select("*")
      .eq("task_id", taskId)
      .order("scan_order", { ascending: true });
    if (data) setScans(data);
  }, [taskId]);

  const fetchExistingReports = useCallback(async () => {
    if (!taskId || !user) return;
    const { data } = await supabase
      .from("hil_reports")
      .select("*")
      .eq("task_id", taskId)
      .eq("doctor_id", user.id);
    if (data) {
      const d: Record<string, ReportDraft> = {};
      const submitted = new Set<string>();
      const rejected: Record<string, string> = {};
      data.forEach((r: any) => {
        d[r.scan_id] = {
          indication: r.indication || "",
          comparison: r.comparison || "",
          findings: r.findings || "",
          impression: r.impression || "",
        };
        if (r.status === "submitted" || r.status === "approved") {
          submitted.add(r.scan_id);
        }
        if (r.status === "rejected") {
          rejected[r.scan_id] = r.admin_feedback || "No specific feedback provided.";
        }
      });
      setDrafts(d);
      setSubmittedScans(submitted);
      setRejectedFeedback(rejected);
    }
    setLoading(false);
  }, [taskId, user]);

  useEffect(() => {
    fetchTask();
    fetchScans();
    fetchExistingReports();
  }, [fetchTask, fetchScans, fetchExistingReports]);

  const currentScan = scans[currentIndex];
  const currentDraft = currentScan
    ? drafts[currentScan.id] || { indication: "Routine chest X-ray", comparison: "No prior studies available", findings: "", impression: "" }
    : null;
  const isCurrentSubmitted = currentScan ? submittedScans.has(currentScan.id) : false;
  const isCurrentRejected = currentScan ? !!rejectedFeedback[currentScan.id] : false;
  const completedCount = submittedScans.size;
  const progressPct = scans.length > 0 ? (completedCount / scans.length) * 100 : 0;

  const updateDraft = (field: keyof ReportDraft, value: string) => {
    if (!currentScan) return;
    setDrafts(prev => ({
      ...prev,
      [currentScan.id]: {
        ...prev[currentScan.id] || { indication: "Routine chest X-ray", comparison: "No prior studies available", findings: "", impression: "" },
        [field]: value,
      }
    }));
  };

  // Auto-save: debounced save after 5s of inactivity
  useEffect(() => {
    if (!currentScan || !currentDraft || isCurrentSubmitted) return;
    const hasContent = currentDraft.findings.trim() || currentDraft.impression.trim();
    if (!hasContent) return;

    if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current);
    autoSaveTimer.current = setTimeout(() => {
      handleSaveDraftSilent();
    }, 5000);

    return () => {
      if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentDraft, currentScan?.id]);

  const handleSaveDraftSilent = async () => {
    if (!currentScan || !currentDraft || !user || !taskId) return;
    try {
      const { data: existing } = await supabase
        .from("hil_reports")
        .select("id")
        .eq("scan_id", currentScan.id)
        .eq("doctor_id", user.id)
        .maybeSingle();

      if (existing) {
        await supabase.from("hil_reports").update({
          indication: currentDraft.indication,
          comparison: currentDraft.comparison,
          findings: currentDraft.findings,
          impression: currentDraft.impression,
          status: "draft",
          updated_at: new Date().toISOString(),
        }).eq("id", existing.id);
      } else {
        await supabase.from("hil_reports").insert({
          scan_id: currentScan.id,
          task_id: taskId,
          doctor_id: user.id,
          indication: currentDraft.indication,
          comparison: currentDraft.comparison,
          findings: currentDraft.findings,
          impression: currentDraft.impression,
          status: "draft",
        });
      }
    } catch {
      // Silent auto-save — don't interrupt the user
    }
  };

  const handleSaveDraft = async () => {
    if (!currentScan || !currentDraft || !user || !taskId) return;
    setSaving(true);
    try {
      const { data: existing } = await supabase
        .from("hil_reports")
        .select("id")
        .eq("scan_id", currentScan.id)
        .eq("doctor_id", user.id)
        .maybeSingle();

      if (existing) {
        await supabase.from("hil_reports").update({
          indication: currentDraft.indication,
          comparison: currentDraft.comparison,
          findings: currentDraft.findings,
          impression: currentDraft.impression,
          status: "draft",
          updated_at: new Date().toISOString(),
        }).eq("id", existing.id);
      } else {
        await supabase.from("hil_reports").insert({
          scan_id: currentScan.id,
          task_id: taskId,
          doctor_id: user.id,
          indication: currentDraft.indication,
          comparison: currentDraft.comparison,
          findings: currentDraft.findings,
          impression: currentDraft.impression,
          status: "draft",
        });
      }
      toast.success("Draft saved!");
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setSaving(false);
    }
  };

  const handleSubmitReport = async () => {
    if (!currentScan || !currentDraft || !user || !taskId) return;
    // Validate all required fields
    if (!currentDraft.indication.trim()) {
      toast.error("Indication is required.");
      return;
    }
    if (!currentDraft.findings.trim() || !currentDraft.impression.trim()) {
      toast.error("Findings and Impression are required.");
      return;
    }
    setSaving(true);
    try {
      const { data: existing } = await supabase
        .from("hil_reports")
        .select("id")
        .eq("scan_id", currentScan.id)
        .eq("doctor_id", user.id)
        .maybeSingle();

      if (existing) {
        await supabase.from("hil_reports").update({
          indication: currentDraft.indication,
          comparison: currentDraft.comparison,
          findings: currentDraft.findings,
          impression: currentDraft.impression,
          status: "submitted",
          updated_at: new Date().toISOString(),
        }).eq("id", existing.id);
      } else {
        await supabase.from("hil_reports").insert({
          scan_id: currentScan.id,
          task_id: taskId,
          doctor_id: user.id,
          indication: currentDraft.indication,
          comparison: currentDraft.comparison,
          findings: currentDraft.findings,
          impression: currentDraft.impression,
          status: "submitted",
        });
      }

      // Update scan status
      await supabase.from("hil_scans").update({ status: "labeled" }).eq("id", currentScan.id);

      // Update local state
      const newSubmitted = new Set(submittedScans);
      newSubmitted.add(currentScan.id);
      setSubmittedScans(newSubmitted);

      // Clear rejection feedback if re-submitting
      if (rejectedFeedback[currentScan.id]) {
        const newRejected = { ...rejectedFeedback };
        delete newRejected[currentScan.id];
        setRejectedFeedback(newRejected);
      }

      // Re-fetch task to get DB-trigger-updated counts
      await fetchTask();

      toast.success(`Report submitted! (${newSubmitted.size}/${scans.length})`);

      // Auto-advance to next unlabeled scan
      if (currentIndex < scans.length - 1) {
        setCurrentIndex(currentIndex + 1);
      }
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex h-[60vh] items-center justify-center">
          <div className="animate-pulse text-muted-foreground">Loading labeling task...</div>
        </div>
      </Layout>
    );
  }

  if (!task) {
    return (
      <Layout>
        <div className="flex h-[60vh] flex-col items-center justify-center gap-4">
          <AlertCircle className="w-12 h-12 text-destructive" />
          <p className="text-muted-foreground">Task not found.</p>
          <Button onClick={() => navigate("/patients")}>Go Back</Button>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="figma-page-shell space-y-6">
        {/* Header */}
        <div className="figma-workspace-hero grid gap-5 lg:grid-cols-[1fr_auto_280px] lg:items-center">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate("/patients")}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <Badge variant="outline" className="eyebrow mb-2">
                Human-in-the-loop review
              </Badge>
              <h1 className="flex items-center gap-2 text-2xl font-extrabold text-foreground md:text-4xl">
                <FileText className="w-6 h-6 text-primary" />
                {task.title}
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Human-in-the-Loop Labeling Task
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="text-sm px-3 py-1">
              {completedCount}/{scans.length} Labeled
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowInstructions(!showInstructions)}
            >
              <Info className="w-4 h-4 mr-1" />
              {showInstructions ? "Hide" : "Show"} Instructions
            </Button>
          </div>
          <RadiologyImageCard
            src={radiologyImages.chestXrayReview}
            alt="Radiologist labeling chest X-ray"
            label="Expert labeling"
            caption="Human feedback for model quality"
            className="h-44"
          />
        </div>

        {/* Progress */}
        <div className="space-y-2">
          <Progress value={progressPct} className="h-2 rounded-full" />
          <p className="text-xs text-muted-foreground text-right">{Math.round(progressPct)}% complete</p>
        </div>

        {/* Instructions Panel */}
        {showInstructions && (
          <Card className="surface-card border-primary/30 bg-primary/5">
            <CardContent className="pt-4 pb-4">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-primary mt-0.5 shrink-0" />
                <div className="space-y-2">
                  <h3 className="font-semibold text-foreground">Instructions</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {task.instructions || "Please review each X-ray scan and fill in the report following the IU X-ray dataset format. Provide detailed findings and a concise impression for each scan."}
                  </p>
                  <div className="grid grid-cols-2 gap-4 mt-3 text-xs text-muted-foreground">
                    <div className="rounded-[20px] border bg-white/75 p-3">
                      <strong className="text-foreground block mb-1">Findings</strong>
                      Describe all observations: lung fields, heart size, mediastinum, bones, and any abnormalities.
                    </div>
                    <div className="rounded-[20px] border bg-white/75 p-3">
                      <strong className="text-foreground block mb-1">Impression</strong>
                      Summarize your diagnostic interpretation concisely (e.g., "No acute cardiopulmonary disease").
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Scan Image */}
          <Card className="surface-card overflow-hidden">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Eye className="w-5 h-5 text-primary" />
                  Scan {currentIndex + 1} of {scans.length}
                </CardTitle>
                {isCurrentSubmitted && (
                  <Badge className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30">
                    <CheckCircle className="w-3 h-3 mr-1" /> Submitted
                  </Badge>
                )}
                {isCurrentRejected && !isCurrentSubmitted && (
                  <Badge className="bg-destructive/10 text-destructive border-destructive/30">
                    <XCircle className="w-3 h-3 mr-1" /> Rejected
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {currentScan && (
                <div className="relative overflow-hidden rounded-[26px] border bg-black/5">
                  <img
                    src={currentScan.image_url}
                    alt={`Scan ${currentIndex + 1}`}
                    className="w-full h-auto max-h-[600px] object-contain mx-auto"
                  />
                </div>
              )}

              {/* Navigation */}
              <div className="flex items-center justify-between mt-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
                  disabled={currentIndex === 0}
                >
                  <ChevronLeft className="w-4 h-4 mr-1" /> Previous
                </Button>

                {/* Scan dots */}
                <div className="flex items-center gap-1.5 flex-wrap justify-center max-w-[60%]">
                  {scans.map((s, i) => (
                    <button
                      key={s.id}
                      onClick={() => setCurrentIndex(i)}
                      className={cn(
                        "w-3 h-3 rounded-full transition-all border",
                        i === currentIndex
                          ? "bg-primary border-primary scale-125"
                          : submittedScans.has(s.id)
                            ? "bg-emerald-500 border-emerald-500"
                            : rejectedFeedback[s.id]
                              ? "bg-destructive border-destructive"
                              : "bg-muted border-border hover:bg-muted-foreground/30"
                      )}
                      title={`Scan ${i + 1}${submittedScans.has(s.id) ? " (submitted)" : rejectedFeedback[s.id] ? " (rejected)" : ""}`}
                    />
                  ))}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentIndex(Math.min(scans.length - 1, currentIndex + 1))}
                  disabled={currentIndex === scans.length - 1}
                >
                  Next <ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Right: Report Form */}
          <Card className="surface-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <FileText className="w-5 h-5 text-primary" />
                Radiology Report
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Rejection Feedback Banner */}
              {isCurrentRejected && !isCurrentSubmitted && currentScan && (
                <div className="space-y-1 rounded-[20px] border border-destructive/30 bg-destructive/10 p-3">
                  <p className="text-sm font-medium text-destructive flex items-center gap-1">
                    <XCircle className="w-4 h-4" /> Report Rejected by Admin
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {rejectedFeedback[currentScan.id]}
                  </p>
                  <p className="text-xs text-muted-foreground italic">
                    Please revise your report and re-submit.
                  </p>
                </div>
              )}

              {/* Indication */}
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  Indication <span className="text-destructive">*</span>
                </label>
                <Input
                  placeholder="Reason for examination (e.g., cough, chest pain, shortness of breath)"
                  value={currentDraft?.indication || ""}
                  onChange={(e) => updateDraft("indication", e.target.value)}
                  disabled={isCurrentSubmitted}
                />
              </div>

              {/* Comparison */}
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  Comparison <span className="text-muted-foreground font-normal">(optional)</span>
                </label>
                <Input
                  placeholder="Prior studies (e.g., No prior studies available)"
                  value={currentDraft?.comparison || ""}
                  onChange={(e) => updateDraft("comparison", e.target.value)}
                  disabled={isCurrentSubmitted}
                />
              </div>

              {/* Findings */}
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  Findings <span className="text-destructive">*</span>
                </label>
                <Textarea
                  placeholder="Describe all observations: lung fields, heart size, mediastinum, bones, pleural spaces, and any abnormalities. Be thorough and specific."
                  value={currentDraft?.findings || ""}
                  onChange={(e) => updateDraft("findings", e.target.value)}
                  disabled={isCurrentSubmitted}
                  className="min-h-[140px] resize-y"
                />
              </div>

              {/* Impression */}
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  Impression <span className="text-destructive">*</span>
                </label>
                <Textarea
                  placeholder="Summarize your diagnostic interpretation (e.g., No acute cardiopulmonary disease. Stable cardiomegaly.)"
                  value={currentDraft?.impression || ""}
                  onChange={(e) => updateDraft("impression", e.target.value)}
                  disabled={isCurrentSubmitted}
                  className="min-h-[100px] resize-y"
                />
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-2">
                <Button
                  variant="outline"
                  className="flex-1 gap-2"
                  onClick={handleSaveDraft}
                  disabled={saving || isCurrentSubmitted}
                >
                  <Save className="w-4 h-4" />
                  Save Draft
                </Button>
                <Button
                  className="flex-1 gap-2"
                  onClick={handleSubmitReport}
                  disabled={saving || isCurrentSubmitted}
                >
                  <Send className="w-4 h-4" />
                  {isCurrentSubmitted ? "Submitted" : isCurrentRejected ? "Re-submit Report" : "Submit Report"}
                </Button>
              </div>

              {isCurrentSubmitted && (
                <div className="rounded-[20px] border border-emerald-500/30 bg-emerald-500/10 p-3 text-center">
                  <p className="text-sm text-emerald-600 flex items-center justify-center gap-1">
                    <CheckCircle className="w-4 h-4" />
                    This report has been submitted and is awaiting admin review.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
};

export default HILLabelingPage;
