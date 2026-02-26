import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import {
  UserCheck,
  Save,
  CheckCircle,
  RotateCcw,
  X,
  Plus,
  Brain,
  Sparkles,
  Stethoscope,
  Tag,
  Network,
  MessageSquarePlus,
  ChevronDown,
  ChevronUp,
  Trash2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAnalysis, type FeedbackStatus } from "@/context/AnalysisContext";
import { supabase } from "@/lib/supabaseClient";
import { toast } from "sonner";

// ---------- Types ----------
interface EditableReport {
  findings: string;
  impression: string;
  recommendation: string;
  labels: string[];
}

interface KGEntity {
  label: string;
  type: string;
  included: boolean;
}

interface FeedbackPanelProps {
  onReAnalyze?: () => void;
}

// ---------- Component ----------
const FeedbackPanel = ({ onReAnalyze }: FeedbackPanelProps) => {
  const { report, knowledgeGraphData, feedbackStatus, setFeedbackStatus } = useAnalysis();

  // Local editable state
  const [edited, setEdited] = useState<EditableReport>({
    findings: "",
    impression: "",
    recommendation: "",
    labels: [],
  });
  const [entities, setEntities] = useState<KGEntity[]>([]);
  const [doctorNotes, setDoctorNotes] = useState("");
  const [newLabel, setNewLabel] = useState("");
  const [saving, setSaving] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    report: true,
    labels: true,
    kg: false,
    notes: true,
  });

  // Populate from context on mount / when report changes
  useEffect(() => {
    if (!report) return;
    const extReport = report as any;
    setEdited({
      findings: report.findings || "",
      impression: report.impression || "",
      recommendation: extReport.recommendation || "",
      labels: [...report.labels],
    });
  }, [report]);

  useEffect(() => {
    if (!knowledgeGraphData?.entities) return;
    const mapped: KGEntity[] = knowledgeGraphData.entities.map(
      ([label, type]: [string, string]) => ({
        label,
        type,
        included: true,
      })
    );
    setEntities(mapped);
  }, [knowledgeGraphData]);

  // --- Helpers ---
  const toggleSection = (key: keyof typeof expandedSections) =>
    setExpandedSections((prev) => ({ ...prev, [key]: !prev[key] }));

  const removeLabel = (idx: number) =>
    setEdited((prev) => ({ ...prev, labels: prev.labels.filter((_, i) => i !== idx) }));

  const addLabel = () => {
    const l = newLabel.trim();
    if (!l || edited.labels.includes(l)) return;
    setEdited((prev) => ({ ...prev, labels: [...prev.labels, l] }));
    setNewLabel("");
  };

  const toggleEntity = (idx: number) =>
    setEntities((prev) =>
      prev.map((e, i) => (i === idx ? { ...e, included: !e.included } : e))
    );

  const removeEntity = (idx: number) =>
    setEntities((prev) => prev.filter((_, i) => i !== idx));

  // --- Save/Approve ---
  const handleSave = async (status: FeedbackStatus) => {
    setSaving(true);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      const payload = {
        doctor_id: user?.id || null,
        original_findings: report?.findings || "",
        edited_findings: edited.findings,
        original_impression: report?.impression || "",
        edited_impression: edited.impression,
        edited_recommendation: edited.recommendation,
        edited_labels: edited.labels,
        edited_kg: entities.filter((e) => e.included).map((e) => ({ label: e.label, type: e.type })),
        doctor_notes: doctorNotes,
        status: status === "approved" ? "approved" : "draft",
      };

      const { error } = await supabase.from("feedback").insert(payload);
      if (error) throw error;

      setFeedbackStatus(status);
      toast.success(
        status === "approved"
          ? "Report approved and saved!"
          : "Draft saved successfully."
      );
    } catch (e: any) {
      toast.error(e.message || "Failed to save feedback.");
    } finally {
      setSaving(false);
    }
  };

  // Already approved
  if (feedbackStatus === "approved") {
    return (
      <Card className="border-primary/30 bg-primary/5 backdrop-blur animate-in fade-in duration-300">
        <CardContent className="py-8">
          <div className="text-center">
            <CheckCircle className="w-12 h-12 mx-auto mb-3 text-primary" />
            <h3 className="text-lg font-bold text-foreground mb-1">Report Approved</h3>
            <p className="text-sm text-muted-foreground">
              This report has been reviewed and approved by the attending physician.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!report) return null;

  // --- Render Collapsible Section ---
  const Section = ({
    id,
    icon: Icon,
    title,
    children,
  }: {
    id: keyof typeof expandedSections;
    icon: any;
    title: string;
    children: React.ReactNode;
  }) => (
    <div className="border border-border/50 rounded-lg overflow-hidden">
      <button
        onClick={() => toggleSection(id)}
        className="w-full flex items-center justify-between px-4 py-3 bg-muted/30 hover:bg-muted/50 transition-colors"
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-foreground">
          <Icon className="w-4 h-4 text-primary" />
          {title}
        </span>
        {expandedSections[id] ? (
          <ChevronUp className="w-4 h-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        )}
      </button>
      {expandedSections[id] && <div className="p-4 space-y-3">{children}</div>}
    </div>
  );

  return (
    <Card className="border-primary/20 shadow-xl animate-in fade-in slide-in-from-bottom-4 duration-500 backdrop-blur">
      <div className="h-1 bg-gradient-to-r from-amber-400 via-primary to-amber-400" />
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <UserCheck className="w-5 h-5 text-primary" />
          Human-in-the-Loop Review
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Review, correct, and approve the AI-generated report before finalizing.
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* ── Report Text Sections ─────────── */}
        <Section id="report" icon={Brain} title="Report Content">
          <div>
            <label className="text-xs font-bold uppercase text-muted-foreground mb-1.5 block">
              Findings
            </label>
            <Textarea
              value={edited.findings}
              onChange={(e) => setEdited((p) => ({ ...p, findings: e.target.value }))}
              rows={5}
              className="bg-background/50 text-sm leading-relaxed resize-y"
            />
          </div>
          <div>
            <label className="text-xs font-bold uppercase text-muted-foreground mb-1.5 block">
              Impression
            </label>
            <Textarea
              value={edited.impression}
              onChange={(e) => setEdited((p) => ({ ...p, impression: e.target.value }))}
              rows={3}
              className="bg-background/50 text-sm leading-relaxed resize-y"
            />
          </div>
          <div>
            <label className="text-xs font-bold uppercase text-muted-foreground mb-1.5 block">
              Recommendation
            </label>
            <Textarea
              value={edited.recommendation}
              onChange={(e) => setEdited((p) => ({ ...p, recommendation: e.target.value }))}
              rows={2}
              className="bg-background/50 text-sm leading-relaxed resize-y"
            />
          </div>
        </Section>

        {/* ── Labels ───────────────────────── */}
        <Section id="labels" icon={Tag} title="Diagnosis Labels">
          <div className="flex flex-wrap gap-2">
            {edited.labels.map((label, i) => (
              <Badge
                key={i}
                className="gap-1 pr-1 bg-primary text-primary-foreground hover:bg-primary/90 cursor-default"
              >
                {label}
                <button
                  onClick={() => removeLabel(i)}
                  className="ml-1 p-0.5 rounded-full hover:bg-white/20 transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </Badge>
            ))}
          </div>
          <div className="flex gap-2">
            <Input
              value={newLabel}
              onChange={(e) => setNewLabel(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addLabel()}
              placeholder="Add label..."
              className="bg-background/50 text-sm flex-1"
            />
            <Button variant="outline" size="sm" onClick={addLabel} className="gap-1">
              <Plus className="w-3 h-3" /> Add
            </Button>
          </div>
        </Section>

        {/* ── KG Entities ──────────────────── */}
        {entities.length > 0 && (
          <Section id="kg" icon={Network} title={`Knowledge Graph Entities (${entities.length})`}>
            <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
              {entities.map((ent, i) => (
                <div
                  key={i}
                  className={cn(
                    "flex items-center justify-between px-3 py-2 rounded-md border transition-all text-sm",
                    ent.included
                      ? "bg-background/50 border-border/50"
                      : "bg-muted/30 border-border/20 opacity-60 line-through"
                  )}
                >
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-[10px] px-1.5">
                      {ent.type}
                    </Badge>
                    <span>{ent.label}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => toggleEntity(i)}
                      title={ent.included ? "Exclude" : "Include"}
                    >
                      {ent.included ? (
                        <CheckCircle className="w-4 h-4 text-primary" />
                      ) : (
                        <RotateCcw className="w-4 h-4 text-muted-foreground" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => removeEntity(i)}
                      title="Remove"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-destructive" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        )}

        {/* ── Doctor Notes ─────────────────── */}
        <Section id="notes" icon={MessageSquarePlus} title="Doctor Notes">
          <Textarea
            value={doctorNotes}
            onChange={(e) => setDoctorNotes(e.target.value)}
            rows={3}
            placeholder="Additional clinical context, corrections rationale, or observations..."
            className="bg-background/50 text-sm resize-y"
          />
        </Section>

        {/* ── Action Buttons ───────────────── */}
        <div className="flex flex-col sm:flex-row gap-3 pt-2">
          <Button
            onClick={() => handleSave("approved")}
            disabled={saving}
            className="flex-1 gap-2 shadow-glow"
          >
            <CheckCircle className="w-4 h-4" />
            {saving ? "Saving..." : "Approve as Final"}
          </Button>
          <Button
            variant="outline"
            onClick={() => handleSave("draft")}
            disabled={saving}
            className="flex-1 gap-2"
          >
            <Save className="w-4 h-4" />
            Save Draft
          </Button>
          {onReAnalyze && (
            <Button
              variant="outline"
              onClick={onReAnalyze}
              disabled={saving}
              className="gap-2 text-destructive hover:text-destructive border-destructive/30 hover:border-destructive/50 hover:bg-destructive/5"
            >
              <RotateCcw className="w-4 h-4" />
              Re-analyze
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default FeedbackPanel;
