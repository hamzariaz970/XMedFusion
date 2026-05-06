import { useState, useEffect, useCallback, useRef } from "react";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Users, Search, Activity, Plus, UserCog, Stethoscope, FileText, Server, Cpu, HardDrive, Clock, Trash2, Pencil, ShieldCheck, ShieldOff, Wifi, WifiOff, Zap, CheckCircle, XCircle, Crown, Brain, Upload, Eye, MessageSquare, Play, Sparkles, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabaseClient";
import { toast } from "sonner";
import { useAuth } from "@/context/AuthContext";
import { getApiBase, getNgrokHeaders } from "@/lib/apiConfig";

interface Doctor { id: string; user_id: string; full_name: string; email: string; specialization: string; status: string; created_at: string; }
interface UserRole { id: string; user_id: string; role: string; approval_status: string; created_at: string; }
interface PendingRequest { doctor: Doctor; role: UserRole; }
interface HealthData { status: string; uptime_seconds: number; cpu_percent: number; memory_used_mb: number; memory_total_mb: number; gpu_available: boolean; gpu_name?: string; gpu_memory_used_mb?: number; gpu_memory_total_mb?: number; }
interface HILTask { id: string; admin_id: string; doctor_id: string; title: string; instructions: string; total_scans: number; completed_scans: number; approved_scans?: number; status: string; created_at: string; }
interface HILReport { id: string; scan_id: string; task_id: string; doctor_id: string; indication: string; comparison: string; findings: string; impression: string; admin_feedback: string; status: string; }
interface HILScan { id: string; task_id: string; image_url: string; scan_order: number; status: string; }
interface FeedbackDoctor {
  user_id: string;
  full_name: string;
  email: string;
}
interface FeedbackScanMeta {
  id: string;
  patient_id?: string | null;
  scan_type?: string | null;
  original_image_url: string | null;
  source_images?: { url: string; filename?: string; order?: number }[] | null;
  heatmap_image_url?: string | null;
  explainability_reference_image_url?: string | null;
  created_at?: string | null;
}
interface FeedbackPatientMeta {
  id: string;
  age: number | null;
  gender: string | null;
}
interface ClinicalFeedbackRow {
  id: string;
  doctor_id: string | null;
  scan_id: string | null;
  original_findings: string | null;
  edited_findings: string | null;
  original_impression: string | null;
  edited_impression: string | null;
  status: string;
  doctor?: FeedbackDoctor | null;
  scan?: FeedbackScanMeta | null;
  patient?: FeedbackPatientMeta | null;
  batchItem?: HILFeedbackBatchItem | null;
}
interface HILFeedbackBatch {
  id: string;
  admin_id: string | null;
  title: string;
  instructions: string | null;
  model_target: string;
  status: string;
  queued_job_id: string | null;
  created_at: string;
  updated_at: string;
}
interface HILFeedbackBatchItem {
  id: string;
  batch_id: string;
  feedback_id: string | null;
  scan_id: string | null;
  doctor_id: string | null;
  source_type?: "feedback" | "hil_report";
  hil_report_id?: string | null;
  hil_scan_id?: string | null;
  item_order: number;
  include_original_report: boolean;
  anonymize_patient: boolean;
  created_at: string;
  updated_at: string;
}
interface HILTrainingJob {
  id: string;
  batch_id: string | null;
  status: string;
  sample_count: number;
  error: string | null;
  result: Record<string, unknown> | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}
interface LegacyHilBatchReportMeta {
  id: string;
  task_id: string;
  doctor_id: string;
  findings: string;
  impression: string;
  status: string;
  scan_id: string;
}
interface LegacyHilBatchScanMeta {
  id: string;
  task_id: string;
  image_url: string;
  scan_order: number;
}

function formatUptime(s: number) { const h = Math.floor(s/3600); const m = Math.floor((s%3600)/60); return h > 0 ? `${h}h ${m}m` : `${m}m`; }
function getErrorMessage(error: unknown) { return error instanceof Error ? error.message : String(error || "Unexpected error"); }

function getPreviewImageUrl(scan?: FeedbackScanMeta | null) {
  if (scan?.explainability_reference_image_url) {
    return scan.explainability_reference_image_url;
  }

  if (Array.isArray(scan?.source_images)) {
    const firstSourceImage = scan.source_images
      .map((image) => String(image?.url || "").trim())
      .find(Boolean);
    if (firstSourceImage) {
      return firstSourceImage;
    }
  }

  const rawUrl = scan?.original_image_url || scan?.heatmap_image_url || "";
  if (!rawUrl) return null;

  const firstUrl = rawUrl
    .split(",")
    .map((part) => part.trim())
    .find(Boolean);

  return firstUrl || null;
}

function deriveHilTaskProgress(task: HILTask, reports: HILReport[]) {
  const taskReports = reports.filter((report) => report.task_id === task.id);
  const labeledScanIds = new Set(
    taskReports
      .filter((report) => report.status === "submitted" || report.status === "approved")
      .map((report) => report.scan_id)
  );
  const approvedScanIds = new Set(
    taskReports
      .filter((report) => report.status === "approved")
      .map((report) => report.scan_id)
  );

  let status = task.status;
  if (labeledScanIds.size >= task.total_scans && task.total_scans > 0) status = "completed";
  else if (labeledScanIds.size > 0) status = "in_progress";
  else status = "assigned";

  return {
    ...task,
    completed_scans: labeledScanIds.size,
    approved_scans: approvedScanIds.size,
    status,
  };
}

const AdminDashboard = () => {
  const { isAdmin } = useAuth();
  const [activeTab, setActiveTab] = useState<"doctors"|"pending"|"admins"|"hil">("doctors");
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [pendingRequests, setPendingRequests] = useState<PendingRequest[]>([]);
  const [adminRoles, setAdminRoles] = useState<UserRole[]>([]);
  const [patientCount, setPatientCount] = useState(0);
  const [reportCount, setReportCount] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [health, setHealth] = useState<HealthData|null>(null);
  const [healthError, setHealthError] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [adminModalOpen, setAdminModalOpen] = useState(false);
  const [editingDoctor, setEditingDoctor] = useState<Doctor|null>(null);
  const [formName, setFormName] = useState("");
  const [formEmail, setFormEmail] = useState("");
  const [formSpec, setFormSpec] = useState("Radiology");
  const [formStatus, setFormStatus] = useState("active");
  const [adminEmail, setAdminEmail] = useState("");
  const [saving, setSaving] = useState(false);
  // HIL state
  const [hilTasks, setHilTasks] = useState<HILTask[]>([]);
  const [hilModalOpen, setHilModalOpen] = useState(false);
  const [hilModalMode, setHilModalMode] = useState<"new"|"existing">("new");
  const [hilAddToTaskId, setHilAddToTaskId] = useState("");
  const [hilTitle, setHilTitle] = useState("");
  const [hilInstructions, setHilInstructions] = useState("");
  const [hilDoctorId, setHilDoctorId] = useState("");
  const [hilFiles, setHilFiles] = useState<File[]>([]);
  const [hilReviewTaskId, setHilReviewTaskId] = useState<string|null>(null);
  const [hilReviewScans, setHilReviewScans] = useState<(HILScan & { report?: HILReport })[]>([]);
  const [hilFinetuning, setHilFinetuning] = useState(false);
  const [clinicalFeedback, setClinicalFeedback] = useState<ClinicalFeedbackRow[]>([]);
  const [selectedFeedbackIds, setSelectedFeedbackIds] = useState<Set<string>>(new Set());
  const [batchApproving, setBatchApproving] = useState(false);
  const [hilFeedbackBatches, setHilFeedbackBatches] = useState<HILFeedbackBatch[]>([]);
  const [hilFeedbackBatchItems, setHilFeedbackBatchItems] = useState<HILFeedbackBatchItem[]>([]);
  const [hilTrainingJobs, setHilTrainingJobs] = useState<HILTrainingJob[]>([]);
  const [legacyHilBatchReports, setLegacyHilBatchReports] = useState<LegacyHilBatchReportMeta[]>([]);
  const [legacyHilBatchScans, setLegacyHilBatchScans] = useState<LegacyHilBatchScanMeta[]>([]);
  const [selectedFeedbackBatchId, setSelectedFeedbackBatchId] = useState<string>("");
  const [feedbackBatchModalOpen, setFeedbackBatchModalOpen] = useState(false);
  const [editingFeedbackBatch, setEditingFeedbackBatch] = useState<HILFeedbackBatch | null>(null);
  const [feedbackBatchTitle, setFeedbackBatchTitle] = useState("");
  const [feedbackBatchInstructions, setFeedbackBatchInstructions] = useState("");
  const [queueingFeedbackBatchId, setQueueingFeedbackBatchId] = useState<string | null>(null);
  const [finetuneConfirmTaskId, setFinetuneConfirmTaskId] = useState<string|null>(null);
  const [finetuneSuccessOpen, setFinetuneSuccessOpen] = useState(false);
  const [finetuneSuccessInfo, setFinetuneSuccessInfo] = useState<{samples:number;taskTitle:string}>({samples:0,taskTitle:""});
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDoctors = useCallback(async () => {
    const { data } = await supabase.from("doctors").select("*").order("created_at", { ascending: false });
    if (data) setDoctors(data);
  }, []);

  const fetchPending = useCallback(async () => {
    const { data: roles } = await supabase.from("user_roles").select("*").eq("approval_status", "pending");
    if (!roles || roles.length === 0) { setPendingRequests([]); return; }
    const userIds = roles.map(r => r.user_id);
    const { data: docs } = await supabase.from("doctors").select("*").in("user_id", userIds);
    const requests: PendingRequest[] = roles.map(role => ({
      role,
      doctor: docs?.find(d => d.user_id === role.user_id) || { id: "", user_id: role.user_id, full_name: "Unknown", email: "N/A", specialization: "N/A", status: "pending", created_at: role.created_at },
    }));
    setPendingRequests(requests);
  }, []);

  const fetchAdmins = useCallback(async () => {
    const { data } = await supabase.from("user_roles").select("*").eq("role", "admin");
    if (data) setAdminRoles(data);
  }, []);

  const fetchCounts = useCallback(async () => {
    const [p, r] = await Promise.all([
      supabase.from("patients").select("id", { count: "exact", head: true }),
      supabase.from("medical_scans").select("id", { count: "exact", head: true }),
    ]);
    setPatientCount(p.count ?? 0);
    setReportCount(r.count ?? 0);
  }, []);

  const fetchHealth = useCallback(async () => {
    try {
      const API_BASE_URL = await getApiBase(true);
      const res = await fetch(`${API_BASE_URL}/api/health?ngrok-skip-browser-warning=1`, { signal: AbortSignal.timeout(10000), headers: getNgrokHeaders(API_BASE_URL) });
      if (res.ok) {
        setHealth(await res.json());
        setHealthError(false);
      } else {
        setHealth(null);
        setHealthError(true);
      }
    } catch {
      setHealth(null);
      setHealthError(true);
    }
  }, []);

  const fetchHilTasks = useCallback(async () => {
    const { data: tasks } = await supabase.from("hil_tasks").select("*").neq("status", "reviewed").order("created_at", { ascending: false });
    if (!tasks || tasks.length === 0) {
      setHilTasks([]);
      return;
    }

    const taskIds = tasks.map((task) => task.id);
    const { data: reports } = await supabase
      .from("hil_reports")
      .select("id, scan_id, task_id, doctor_id, indication, comparison, findings, impression, admin_feedback, status")
      .in("task_id", taskIds);

    const derivedTasks = tasks.map((task) => deriveHilTaskProgress(task, reports || []));
    setHilTasks(derivedTasks);
  }, []);

  const fetchClinicalFeedback = useCallback(async () => {
    const { data: fbData } = await supabase
      .from("feedback")
      .select("id, doctor_id, scan_id, original_findings, edited_findings, original_impression, edited_impression, status")
      .eq("status", "approved")
      .order("created_at", { ascending: false });

    if (fbData && fbData.length > 0) {
      const scanIds = fbData.map((f) => f.scan_id).filter(Boolean);
      const doctorIds = fbData.map((f) => f.doctor_id).filter(Boolean);
      const feedbackIds = fbData.map((f) => f.id);
      let scans: FeedbackScanMeta[] = [];
      let feedbackDoctors: FeedbackDoctor[] = [];
      let patients: FeedbackPatientMeta[] = [];
      let batchItems: HILFeedbackBatchItem[] = [];

      if (scanIds.length > 0) {
        const { data: scanData } = await supabase
          .from("medical_scans")
          .select("id, patient_id, scan_type, original_image_url, source_images, heatmap_image_url, explainability_reference_image_url, created_at")
          .in("id", scanIds);
        if (scanData) scans = scanData;

        const patientIds = scans.map((scan) => scan.patient_id).filter(Boolean) as string[];
        if (patientIds.length > 0) {
          const { data: patientData } = await supabase
            .from("patients")
            .select("id, age, gender")
            .in("id", patientIds);
          if (patientData) patients = patientData;
        }
      }

      if (doctorIds.length > 0) {
        const { data: doctorData } = await supabase
          .from("doctors")
          .select("user_id, full_name, email")
          .in("user_id", doctorIds);
        if (doctorData) feedbackDoctors = doctorData;
      }

      const { data: batchItemData } = await supabase
        .from("hil_feedback_batch_items")
        .select("*")
        .in("feedback_id", feedbackIds);
      if (batchItemData) batchItems = batchItemData;

      const merged: ClinicalFeedbackRow[] = fbData.map((f) => ({
        ...f,
        doctor: feedbackDoctors.find((doctor) => doctor.user_id === f.doctor_id) || null,
        scan: scans.find((scan) => scan.id === f.scan_id) || null,
        patient: patients.find((patient) => patient.id === scans.find((scan) => scan.id === f.scan_id)?.patient_id) || null,
        batchItem: batchItems.find((item) => item.feedback_id === f.id) || null,
      }));
      setClinicalFeedback(merged);
    } else {
      setClinicalFeedback([]);
    }
  }, []);

  const fetchHilFeedbackBatches = useCallback(async () => {
    const [{ data: batches, error: batchError }, { data: items }, { data: jobs }] = await Promise.all([
      supabase.from("hil_feedback_batches").select("*").order("created_at", { ascending: false }),
      supabase.from("hil_feedback_batch_items").select("*").order("item_order", { ascending: true }),
      supabase.from("hil_training_jobs").select("*").order("created_at", { ascending: false }).limit(20),
    ]);

    if (batchError) {
      console.warn("HIL feedback batches unavailable:", batchError.message);
      setHilFeedbackBatches([]);
      setHilFeedbackBatchItems([]);
      setHilTrainingJobs([]);
      return;
    }

    setHilFeedbackBatches(batches || []);
    setHilFeedbackBatchItems(items || []);
    setHilTrainingJobs(jobs || []);

    const hilReportIds = (items || []).map((item) => item.hil_report_id).filter(Boolean) as string[];
    const hilScanIds = (items || []).map((item) => item.hil_scan_id).filter(Boolean) as string[];
    if (hilReportIds.length > 0) {
      const { data: legacyReports } = await supabase
        .from("hil_reports")
        .select("id, task_id, doctor_id, findings, impression, status, scan_id")
        .in("id", hilReportIds);
      setLegacyHilBatchReports(legacyReports || []);
    } else {
      setLegacyHilBatchReports([]);
    }
    if (hilScanIds.length > 0) {
      const { data: legacyScans } = await supabase
        .from("hil_scans")
        .select("id, task_id, image_url, scan_order")
        .in("id", hilScanIds);
      setLegacyHilBatchScans(legacyScans || []);
    } else {
      setLegacyHilBatchScans([]);
    }

    if (!selectedFeedbackBatchId && batches && batches.length > 0) {
      setSelectedFeedbackBatchId(batches[0].id);
    }
  }, [selectedFeedbackBatchId]);

  useEffect(() => {
    fetchDoctors(); fetchPending(); fetchAdmins(); fetchCounts(); fetchHealth(); fetchHilTasks(); fetchHilFeedbackBatches(); fetchClinicalFeedback();
    const interval = setInterval(fetchHealth, 10000);

    // Real-time subscription: refresh clinical feedback whenever a doctor inserts/updates a feedback row
    const channel = supabase
      .channel("feedback-admin-watch")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "feedback" },
        () => { fetchClinicalFeedback(); fetchHilFeedbackBatches(); }
      )
      .subscribe();

    return () => {
      clearInterval(interval);
      supabase.removeChannel(channel);
    };
  }, [fetchDoctors, fetchPending, fetchAdmins, fetchCounts, fetchHealth, fetchHilTasks, fetchHilFeedbackBatches, fetchClinicalFeedback]);

  // Approve / Reject pending
  const handleApprove = async (req: PendingRequest) => {
    const { error: e1 } = await supabase.from("user_roles").update({ approval_status: "approved" }).eq("id", req.role.id);
    const { error: e2 } = await supabase.from("doctors").update({ status: "active" }).eq("user_id", req.role.user_id);
    if (e1 || e2) toast.error("Failed to approve"); else { toast.success(`Dr. ${req.doctor.full_name} approved!`); fetchPending(); fetchDoctors(); }
  };

  const handleReject = async (req: PendingRequest) => {
    if (!confirm(`Reject Dr. ${req.doctor.full_name}?`)) return;
    const { error } = await supabase.from("user_roles").update({ approval_status: "rejected" }).eq("id", req.role.id);
    if (error) toast.error("Failed to reject"); else { toast.success("Request rejected."); fetchPending(); }
  };

  // Doctor CRUD
  const openAdd = () => { setEditingDoctor(null); setFormName(""); setFormEmail(""); setFormSpec("Radiology"); setFormStatus("active"); setModalOpen(true); };
  const openEdit = (doc: Doctor) => { setEditingDoctor(doc); setFormName(doc.full_name); setFormEmail(doc.email); setFormSpec(doc.specialization); setFormStatus(doc.status); setModalOpen(true); };

  const handleSave = async () => {
    if (!formName.trim() || !formEmail.trim()) { toast.error("Name and Email required."); return; }
    setSaving(true);
    try {
      if (editingDoctor) {
        const { error } = await supabase.from("doctors").update({ full_name: formName.trim(), email: formEmail.trim(), specialization: formSpec, status: formStatus }).eq("id", editingDoctor.id);
        if (error) throw error;
        toast.success("Doctor updated.");
      } else {
        // Admin pre-approves this doctor — when they sign up, they'll be auto-approved
        const { error } = await supabase.from("doctors").insert({ full_name: formName.trim(), email: formEmail.trim(), specialization: formSpec, status: "pre-approved", user_id: crypto.randomUUID() });
        if (error) throw error;
        toast.success("Doctor pre-approved. They will have instant access when they sign up.");
      }
      setModalOpen(false); fetchDoctors();
    } catch (e: unknown) { toast.error(getErrorMessage(e)); } finally { setSaving(false); }
  };

  const handleDelete = async (doc: Doctor) => {
    if (!confirm(`Remove Dr. ${doc.full_name}?`)) return;
    const { error } = await supabase.from("doctors").delete().eq("id", doc.id);
    if (error) toast.error(error.message); else { toast.success("Doctor removed."); fetchDoctors(); }
  };

  const toggleStatus = async (doc: Doctor) => {
    const newS = doc.status === "active" ? "suspended" : "active";
    const { error } = await supabase.from("doctors").update({ status: newS }).eq("id", doc.id);
    if (error) toast.error(error.message); else { toast.success(`Doctor ${newS}.`); fetchDoctors(); }
  };

  // Add admin
  const handleAddAdmin = async () => {
    if (!adminEmail.trim()) { toast.error("Email required."); return; }
    setSaving(true);
    try {
      // Look up user by email in doctors table
      const { data: doc } = await supabase.from("doctors").select("user_id").eq("email", adminEmail.trim()).maybeSingle();
      if (!doc) { toast.error("No doctor found with that email. They must register first."); setSaving(false); return; }
      // Update their role to admin
      const { error } = await supabase.from("user_roles").update({ role: "admin", approval_status: "approved" }).eq("user_id", doc.user_id);
      if (error) throw error;
      toast.success(`${adminEmail} promoted to admin!`);
      setAdminModalOpen(false); setAdminEmail(""); fetchAdmins(); fetchPending();
    } catch (e: unknown) { toast.error(getErrorMessage(e)); } finally { setSaving(false); }
  };

  const filtered = doctors.filter(d => {
    const ms = d.full_name.toLowerCase().includes(searchTerm.toLowerCase()) || d.email.toLowerCase().includes(searchTerm.toLowerCase());
    const mf = statusFilter === "all" || d.status === statusFilter;
    return ms && mf;
  });
  const activeDoctors = doctors.filter(d => d.status === "active").length;
  const approvedDoctors = doctors.filter(d => d.status === "active" && d.user_id);
  const selectedFeedbackBatch = hilFeedbackBatches.find((batch) => batch.id === selectedFeedbackBatchId) || null;
  const selectedFeedbackBatchItems = hilFeedbackBatchItems
    .filter((item) => item.batch_id === selectedFeedbackBatchId)
    .sort((a, b) => a.item_order - b.item_order);
  const availableClinicalFeedback = clinicalFeedback.filter((fb) => !fb.batchItem);

  const getFeedbackById = (feedbackId: string) => clinicalFeedback.find((fb) => fb.id === feedbackId);
  const getLegacyHilReportById = (reportId: string) => legacyHilBatchReports.find((report) => report.id === reportId);
  const getLegacyHilScanById = (scanId: string) => legacyHilBatchScans.find((scan) => scan.id === scanId);
  const getBatchItemCount = (batchId: string) => hilFeedbackBatchItems.filter((item) => item.batch_id === batchId).length;
  const getBatchLatestJob = (batchId: string) => hilTrainingJobs.find((job) => job.batch_id === batchId);

  // HIL handlers
  const handleCreateHilTask = async () => {
    if (!hilTitle.trim() || !hilDoctorId || hilFiles.length === 0) { toast.error("Title, doctor, and scans are required."); return; }
    setSaving(true);
    try {
      const { data: task, error: taskErr } = await supabase.from("hil_tasks").insert({
        admin_id: user?.id, doctor_id: hilDoctorId, title: hilTitle.trim(),
        instructions: hilInstructions.trim(), total_scans: hilFiles.length, status: "assigned",
      }).select().single();
      if (taskErr || !task) throw taskErr || new Error("Failed to create task");
      for (let i = 0; i < hilFiles.length; i++) {
        const file = hilFiles[i];
        const filePath = `hil/${task.id}/${i}_${file.name}`;
        const { error: upErr } = await supabase.storage.from("medical-images").upload(filePath, file);
        if (upErr) { console.error("Upload error:", upErr); continue; }
        const { data: urlData } = supabase.storage.from("medical-images").getPublicUrl(filePath);
        await supabase.from("hil_scans").insert({ task_id: task.id, image_url: urlData.publicUrl, scan_order: i });
      }
      toast.success(`Task created with ${hilFiles.length} scans!`);
      setHilModalOpen(false); setHilTitle(""); setHilInstructions(""); setHilDoctorId(""); setHilFiles([]);
      fetchHilTasks();
    } catch (e: unknown) { toast.error(getErrorMessage(e)); } finally { setSaving(false); }
  };

  const handleAddScansToTask = async () => {
    if (!hilAddToTaskId || hilFiles.length === 0) { toast.error("Please select a batch and at least one scan."); return; }
    setSaving(true);
    try {
      // Get the current max scan_order for this task
      const { data: existingScans } = await supabase
        .from("hil_scans").select("scan_order").eq("task_id", hilAddToTaskId).order("scan_order", { ascending: false }).limit(1);
      const startOrder = existingScans && existingScans.length > 0 ? (existingScans[0].scan_order + 1) : 0;

      for (let i = 0; i < hilFiles.length; i++) {
        const file = hilFiles[i];
        const filePath = `hil/${hilAddToTaskId}/${Date.now()}_${i}_${file.name}`;
        const { error: upErr } = await supabase.storage.from("medical-images").upload(filePath, file);
        if (upErr) { console.error("Upload error:", upErr); continue; }
        const { data: urlData } = supabase.storage.from("medical-images").getPublicUrl(filePath);
        await supabase.from("hil_scans").insert({ task_id: hilAddToTaskId, image_url: urlData.publicUrl, scan_order: startOrder + i });
      }

      // Update total_scans count on the task
      const task = hilTasks.find(t => t.id === hilAddToTaskId);
      if (task) {
        await supabase.from("hil_tasks").update({
          total_scans: task.total_scans + hilFiles.length,
          status: "assigned", // re-open if completed
        }).eq("id", hilAddToTaskId);
      }

      toast.success(`${hilFiles.length} scan${hilFiles.length > 1 ? "s" : ""} added to batch!`);
      setHilModalOpen(false); setHilFiles([]); setHilAddToTaskId("");
      fetchHilTasks();
    } catch (e: unknown) { toast.error(getErrorMessage(e)); } finally { setSaving(false); }
  };

  const openHilReview = async (taskId: string) => {
    const { data: scans } = await supabase.from("hil_scans").select("*").eq("task_id", taskId).order("scan_order");
    // Fetch ALL reports (not just submitted) so admin can see full history
    const { data: reports } = await supabase.from("hil_reports").select("*").eq("task_id", taskId);
    if (scans) {
      const merged = scans.map((s: HILScan) => ({ ...s, report: reports?.find((r: HILReport) => r.scan_id === s.id) }));
      setHilReviewScans(merged);
    }
    setHilReviewTaskId(taskId);
    fetchHilTasks();
  };

  const handleApproveReport = async (reportId: string, scanId: string) => {
    await supabase.from("hil_reports").update({ status: "approved" }).eq("id", reportId);
    await supabase.from("hil_scans").update({ status: "approved" }).eq("id", scanId);
    toast.success("Report approved!"); if (hilReviewTaskId) openHilReview(hilReviewTaskId); fetchHilTasks();
  };

  const handleRejectReport = async (reportId: string, scanId: string) => {
    const feedback = prompt("Rejection feedback (optional):") || "";
    await supabase.from("hil_reports").update({ status: "rejected", admin_feedback: feedback }).eq("id", reportId);
    await supabase.from("hil_scans").update({ status: "rejected" }).eq("id", scanId);
    toast.info("Report rejected."); if (hilReviewTaskId) openHilReview(hilReviewTaskId); fetchHilTasks();
  };

  const handleDeleteTask = async (taskId: string) => {
    if (!confirm("Delete this task and all its scans/reports? This cannot be undone.")) return;
    const { error } = await supabase.from("hil_tasks").delete().eq("id", taskId);
    if (error) toast.error(error.message);
    else { toast.success("Task deleted."); fetchHilTasks(); }
  };

  const handleAddApprovedHilReportToBatch = async (scan: HILScan & { report?: HILReport }) => {
    if (!scan.report || scan.report.status !== "approved") {
      toast.error("Approve the report before adding it to the Vision training pool.");
      return;
    }
    if (!selectedFeedbackBatchId) {
      toast.error("Select or create a Vision HIL batch first.");
      return;
    }

    setSaving(true);
    try {
      const existingItem = hilFeedbackBatchItems.find((item) => item.hil_report_id === scan.report?.id);
      const itemOrder = getBatchItemCount(selectedFeedbackBatchId);
      if (existingItem) {
        const { error } = await supabase
          .from("hil_feedback_batch_items")
          .update({
            batch_id: selectedFeedbackBatchId,
            source_type: "hil_report",
            hil_report_id: scan.report.id,
            hil_scan_id: scan.id,
            doctor_id: scan.report.doctor_id,
            item_order: itemOrder,
          })
          .eq("id", existingItem.id);
        if (error) throw error;
      } else {
        const { error } = await supabase.from("hil_feedback_batch_items").insert({
          batch_id: selectedFeedbackBatchId,
          feedback_id: null,
          scan_id: null,
          doctor_id: scan.report.doctor_id,
          source_type: "hil_report",
          hil_report_id: scan.report.id,
          hil_scan_id: scan.id,
          item_order: itemOrder,
          include_original_report: true,
          anonymize_patient: true,
        });
        if (error) throw error;
      }
      toast.success("Approved HIL report added to Vision training batch.");
      await fetchHilFeedbackBatches();
    } catch (e: unknown) {
      toast.error(getErrorMessage(e) || "Failed to add approved HIL report.");
    } finally {
      setSaving(false);
    }
  };

  const handleApproveClinicalHIL = async (feedbackId: string) => {
    // Optimistic update: remove from UI immediately
    setClinicalFeedback(prev => prev.filter(fb => fb.id !== feedbackId));
    setSelectedFeedbackIds(prev => { const n = new Set(prev); n.delete(feedbackId); return n; });
    const { error } = await supabase.from("feedback").update({ status: "hil_approved" }).eq("id", feedbackId);
    if (!error) {
       toast.success("Sent for training the model!");
    } else {
       toast.error(error.message);
       fetchClinicalFeedback();
    }
  };

  const handleBatchApproveClinicalHIL = async () => {
    if (selectedFeedbackIds.size === 0) return;
    setBatchApproving(true);
    const ids = Array.from(selectedFeedbackIds);
    // Optimistic update
    setClinicalFeedback(prev => prev.filter(fb => !selectedFeedbackIds.has(fb.id)));
    setSelectedFeedbackIds(new Set());
    const { error } = await supabase
      .from("feedback")
      .update({ status: "hil_approved" })
      .in("id", ids);
    if (!error) {
      toast.success(`${ids.length} report${ids.length > 1 ? "s" : ""} approved for HIL training!`);
    } else {
      toast.error(error.message);
      fetchClinicalFeedback();
    }
    setBatchApproving(false);
  };

  const toggleFeedbackSelection = (id: string) => {
    setSelectedFeedbackIds(prev => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  };

  const toggleSelectAll = () => {
    if (selectedFeedbackIds.size === availableClinicalFeedback.length) {
      setSelectedFeedbackIds(new Set());
    } else {
      setSelectedFeedbackIds(new Set(availableClinicalFeedback.map(fb => fb.id)));
    }
  };

  const openCreateFeedbackBatch = () => {
    setEditingFeedbackBatch(null);
    setFeedbackBatchTitle("");
    setFeedbackBatchInstructions("");
    setFeedbackBatchModalOpen(true);
  };

  const openEditFeedbackBatch = (batch: HILFeedbackBatch) => {
    setEditingFeedbackBatch(batch);
    setFeedbackBatchTitle(batch.title);
    setFeedbackBatchInstructions(batch.instructions || "");
    setFeedbackBatchModalOpen(true);
  };

  const handleSaveFeedbackBatch = async () => {
    if (!feedbackBatchTitle.trim()) {
      toast.error("Batch name is required.");
      return;
    }
    setSaving(true);
    try {
      if (editingFeedbackBatch) {
        const { error } = await supabase
          .from("hil_feedback_batches")
          .update({ title: feedbackBatchTitle.trim(), instructions: feedbackBatchInstructions.trim() })
          .eq("id", editingFeedbackBatch.id);
        if (error) throw error;
        toast.success("HIL batch updated.");
      } else {
        const { data: batch, error } = await supabase
          .from("hil_feedback_batches")
          .insert({
            admin_id: user?.id,
            title: feedbackBatchTitle.trim(),
            instructions: feedbackBatchInstructions.trim(),
            model_target: "vision_agent",
            status: "draft",
          })
          .select()
          .single();
        if (error || !batch) throw error || new Error("Failed to create batch");
        setSelectedFeedbackBatchId(batch.id);
        if (selectedFeedbackIds.size > 0) {
          await addFeedbackToBatch(batch.id, Array.from(selectedFeedbackIds));
        }
        toast.success("HIL batch created.");
      }
      setFeedbackBatchModalOpen(false);
      setEditingFeedbackBatch(null);
      setFeedbackBatchTitle("");
      setFeedbackBatchInstructions("");
      setSelectedFeedbackIds(new Set());
      await fetchHilFeedbackBatches();
      await fetchClinicalFeedback();
    } catch (e: unknown) {
      toast.error(getErrorMessage(e) || "Failed to save HIL batch.");
    } finally {
      setSaving(false);
    }
  };

  const addFeedbackToBatch = async (batchId: string, feedbackIds: string[]) => {
    if (!batchId || feedbackIds.length === 0) return;
    const startOrder = getBatchItemCount(batchId);
    for (let index = 0; index < feedbackIds.length; index++) {
      const fb = getFeedbackById(feedbackIds[index]);
      if (!fb) continue;
      const existing = fb.batchItem || hilFeedbackBatchItems.find((item) => item.feedback_id === fb.id);
      if (existing) {
        const { error } = await supabase
          .from("hil_feedback_batch_items")
          .update({
            batch_id: batchId,
            scan_id: fb.scan_id,
            doctor_id: fb.doctor_id,
            item_order: startOrder + index,
          })
          .eq("id", existing.id);
        if (error) throw error;
      } else {
        const { error } = await supabase.from("hil_feedback_batch_items").insert({
          batch_id: batchId,
          feedback_id: fb.id,
          scan_id: fb.scan_id,
          doctor_id: fb.doctor_id,
          item_order: startOrder + index,
          include_original_report: true,
          anonymize_patient: true,
        });
        if (error) throw error;
      }
    }
  };

  const handleAddSelectedToFeedbackBatch = async () => {
    if (!selectedFeedbackBatchId || selectedFeedbackIds.size === 0) {
      toast.error("Select a batch and at least one feedback example.");
      return;
    }
    setSaving(true);
    try {
      await addFeedbackToBatch(selectedFeedbackBatchId, Array.from(selectedFeedbackIds));
      toast.success(`${selectedFeedbackIds.size} example${selectedFeedbackIds.size === 1 ? "" : "s"} added to batch.`);
      setSelectedFeedbackIds(new Set());
      await fetchHilFeedbackBatches();
      await fetchClinicalFeedback();
    } catch (e: unknown) {
      toast.error(getErrorMessage(e) || "Failed to add examples to batch.");
    } finally {
      setSaving(false);
    }
  };

  const handleMoveBatchItem = async (itemId: string, batchId: string) => {
    const { error } = await supabase
      .from("hil_feedback_batch_items")
      .update({ batch_id: batchId, item_order: getBatchItemCount(batchId) })
      .eq("id", itemId);
    if (error) toast.error(error.message);
    else {
      toast.success("Example moved.");
      fetchHilFeedbackBatches();
      fetchClinicalFeedback();
    }
  };

  const handleRemoveBatchItem = async (itemId: string) => {
    const { error } = await supabase.from("hil_feedback_batch_items").delete().eq("id", itemId);
    if (error) toast.error(error.message);
    else {
      toast.success("Example returned to available pool.");
      fetchHilFeedbackBatches();
      fetchClinicalFeedback();
    }
  };

  const handleDeleteFeedbackFromPool = async (feedbackId: string, itemId?: string) => {
    if (!confirm("Remove this example from the HIL pool? This will exclude it from batching and training.")) return;
    if (itemId) {
      await supabase.from("hil_feedback_batch_items").delete().eq("id", itemId);
    }
    const { error } = await supabase.from("feedback").update({ status: "hil_removed" }).eq("id", feedbackId);
    if (error) toast.error(error.message);
    else {
      toast.success("Example removed from HIL pool.");
      setSelectedFeedbackIds(prev => { const n = new Set(prev); n.delete(feedbackId); return n; });
      fetchHilFeedbackBatches();
      fetchClinicalFeedback();
    }
  };

  const handleDeleteFeedbackBatch = async (batchId: string) => {
    if (!confirm("Delete this HIL batch? Its examples will return to the available pool.")) return;
    const { error } = await supabase.from("hil_feedback_batches").delete().eq("id", batchId);
    if (error) toast.error(error.message);
    else {
      toast.success("HIL batch deleted.");
      if (selectedFeedbackBatchId === batchId) setSelectedFeedbackBatchId("");
      fetchHilFeedbackBatches();
      fetchClinicalFeedback();
    }
  };

  const handleQueueFeedbackBatch = async (batchId: string) => {
    const count = getBatchItemCount(batchId);
    if (count < 3) {
      toast.error("Use at least 3 examples for a light Vision Agent adaptation.");
      return;
    }
    setQueueingFeedbackBatchId(batchId);
    try {
      const apiBase = await getApiBase();
      const { data: sessionData } = await supabase.auth.getSession();
      const token = sessionData.session?.access_token;
      const res = await fetch(`${apiBase}/api/hil/vision-finetune-batches/${batchId}/enqueue?ngrok-skip-browser-warning=1`, {
        method: "POST",
        headers: {
          ...getNgrokHeaders(apiBase),
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || data.detail || "Failed to queue Vision fine-tuning.");
      toast.success(`Vision fine-tuning queued with ${data.sample_count} sample${data.sample_count === 1 ? "" : "s"}.`);
      await fetchHilFeedbackBatches();
    } catch (e: unknown) {
      toast.error(getErrorMessage(e) || "Failed to queue Vision fine-tuning.");
    } finally {
      setQueueingFeedbackBatchId(null);
    }
  };

  const handleRunFinetune = async (taskId: string) => {
    // Open the confirm dialog instead of browser confirm()
    setFinetuneConfirmTaskId(taskId);
  };

  const doRunFinetune = async (taskId: string) => {
    setFinetuneConfirmTaskId(null);
    const task = hilTasks.find(t => t.id === taskId);
    setHilFinetuning(true);
    try {
      const apiBase = await getApiBase();
      const { data: sessionData } = await supabase.auth.getSession();
      const token = sessionData.session?.access_token;
      const res = await fetch(`${apiBase}/api/hil/finetune?ngrok-skip-browser-warning=1`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getNgrokHeaders(apiBase),
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ task_id: taskId }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || data.detail || "Failed to start fine-tuning.");
      await supabase.from("hil_tasks").update({ status: "reviewed" }).eq("id", taskId);
      setFinetuneSuccessInfo({ samples: data.num_samples ?? task?.completed_scans ?? 0, taskTitle: task?.title ?? "Batch" });
      setHilTasks(prev => prev.filter(t => t.id !== taskId));
      setFinetuneSuccessOpen(true);
      pollFinetuneStatus();
    } catch (e: unknown) {
      setHilFinetuning(false);
      toast.error(getErrorMessage(e) || "Failed to start fine-tuning.");
      fetchHilTasks();
    }
  };

  const pollFinetuneStatus = () => {
    const interval = setInterval(async () => {
      try {
        const apiBase = await getApiBase();
        const res = await fetch(`${apiBase}/api/hil/finetune-status?ngrok-skip-browser-warning=1`, { headers: getNgrokHeaders(apiBase) });
        const data = await res.json();
        if (!data.running) {
          clearInterval(interval);
          setHilFinetuning(false);
          if (data.last_result?.status === "complete") {
            toast.success(`Fine-tuning complete! Loss: ${data.last_result.best_loss}, Samples: ${data.last_result.num_samples}${data.last_result.model_reloaded ? " — Model reloaded ✓" : ""}`);
          } else if (data.last_result?.error) {
            toast.error(`Fine-tuning failed: ${data.last_result.error}`);
          }
          fetchHilTasks();
        }
      } catch { /* ignore polling errors */ }
    }, 3000);
  };

  const { user } = useAuth();

  const tabs = [
    { id: "doctors" as const, label: "Doctor Registry", icon: Stethoscope },
    { id: "pending" as const, label: `Pending Requests (${pendingRequests.length})`, icon: Clock },
    { id: "admins" as const, label: "Admin Management", icon: Crown },
    { id: "hil" as const, label: `HIL Feedback (${hilTasks.length})${availableClinicalFeedback.length > 0 ? ` · ${availableClinicalFeedback.length} available` : ""}`, icon: Brain },
  ];

  return (
    <Layout>
      <div className="figma-page-shell space-y-8">
        {/* Header */}
        <div className="figma-workspace-hero grid gap-5 lg:grid-cols-[1fr_auto_280px] lg:items-center">
          <div>
            <Badge variant="outline" className="eyebrow mb-3">
              <Sparkles className="h-3.5 w-3.5" />
              Platform command center
            </Badge>
            <h1 className="mb-2 flex items-center gap-3 text-3xl font-extrabold tracking-tight text-foreground md:text-5xl">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/15 text-primary shadow-sm"><ShieldCheck className="h-6 w-6" /></div>
              Admin <span className="text-primary">Dashboard</span>
            </h1>
            <p className="text-muted-foreground">Platform oversight — manage doctors, approve requests &amp; monitor health</p>
          </div>
          <div className="flex gap-2">
            <Button className="gap-2 shadow-glow" onClick={openAdd}><Plus className="w-4 h-4" />Add Doctor</Button>
            <Button variant="outline" className="gap-2" onClick={() => setAdminModalOpen(true)}><Crown className="w-4 h-4" />Add Admin</Button>
          </div>
          <RadiologyImageCard
            src={radiologyImages.teamReview}
            alt="Radiology operations review"
            label="Admin oversight"
            caption="Doctors, tasks, and model health"
            className="h-44"
            scanLine={false}
          />
        </div>

        {/* Stat Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          <StatCard icon={Stethoscope} label="Registered Doctors" value={doctors.length} />
          <StatCard icon={Users} label="Active Doctors" value={activeDoctors} />
          <StatCard icon={Clock} label="Pending Requests" value={pendingRequests.length} />
          <StatCard icon={UserCog} label="Patient Records" value={patientCount} />
          <StatCard icon={FileText} label="Total Reports" value={reportCount} />
        </div>

        {/* Tabs */}
        <div className="figma-tool-panel flex gap-2 overflow-x-auto p-2">
          {tabs.map(t => (
            <Button key={t.id} variant={activeTab === t.id ? "default" : "ghost"} size="sm" className={cn("gap-2", activeTab === t.id && "shadow-glow")} onClick={() => setActiveTab(t.id)}>
              <t.icon className="w-4 h-4" />{t.label}
            </Button>
          ))}
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            {/* Doctors Tab */}
            {activeTab === "doctors" && (
              <Card className="surface-card">
                <CardHeader>
                  <div className="flex flex-col md:flex-row md:items-center gap-4">
                    <CardTitle className="flex items-center gap-2"><Stethoscope className="w-5 h-5 text-primary" />Doctor Registry</CardTitle>
                    <div className="flex-1 flex flex-col md:flex-row gap-2">
                      <div className="relative flex-1"><Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" /><Input placeholder="Search doctors..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} className="bg-white/70 pl-9" /></div>
                      <div className="flex gap-2">{["all","active","suspended"].map(s => (<Button key={s} variant={statusFilter===s?"default":"outline"} size="sm" onClick={() => setStatusFilter(s)} className="capitalize">{s==="all"?"All":s}</Button>))}</div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader><TableRow><TableHead>Doctor</TableHead><TableHead>Specialization</TableHead><TableHead>Status</TableHead><TableHead>Joined</TableHead><TableHead className="text-right">Actions</TableHead></TableRow></TableHeader>
                      <TableBody>
                        {filtered.length === 0 ? (
                          <TableRow><TableCell colSpan={5} className="text-center py-8 text-muted-foreground">{doctors.length===0?"No doctors registered yet.":"No matching doctors found."}</TableCell></TableRow>
                        ) : filtered.map(doc => (
                          <TableRow key={doc.id} className="group">
                            <TableCell><div className="flex items-center gap-3"><div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center"><span className="text-sm font-medium text-primary">{doc.full_name.split(" ").map(n=>n[0]).join("")}</span></div><div><p className="font-medium text-foreground">{doc.full_name}</p><p className="text-xs text-muted-foreground">{doc.email}</p></div></div></TableCell>
                            <TableCell><Badge variant="secondary">{doc.specialization}</Badge></TableCell>
                            <TableCell><Badge variant="outline" className={cn("gap-1", doc.status==="active"?"bg-primary/20 text-primary border-primary/30":"bg-destructive/20 text-destructive border-destructive/30")}>{doc.status==="active"?<ShieldCheck className="w-3 h-3"/>:<ShieldOff className="w-3 h-3"/>}{doc.status==="active"?"Active":"Suspended"}</Badge></TableCell>
                            <TableCell className="text-muted-foreground text-sm">{new Date(doc.created_at).toLocaleDateString()}</TableCell>
                            <TableCell><div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity"><Button variant="ghost" size="icon" onClick={() => openEdit(doc)} title="Edit"><Pencil className="w-4 h-4" /></Button><Button variant="ghost" size="icon" onClick={() => toggleStatus(doc)} title={doc.status==="active"?"Suspend":"Activate"}>{doc.status==="active"?<ShieldOff className="w-4 h-4 text-warning"/>:<ShieldCheck className="w-4 h-4 text-primary"/>}</Button><Button variant="ghost" size="icon" onClick={() => handleDelete(doc)} title="Remove"><Trash2 className="w-4 h-4 text-destructive" /></Button></div></TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Pending Tab */}
            {activeTab === "pending" && (
              <Card className="surface-card">
                <CardHeader><CardTitle className="flex items-center gap-2"><Clock className="w-5 h-5 text-amber-500" />Pending Registration Requests</CardTitle></CardHeader>
                <CardContent>
                  {pendingRequests.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground"><CheckCircle className="w-12 h-12 mx-auto mb-3 text-primary/40" /><p>No pending requests. All caught up!</p></div>
                  ) : (
                    <div className="space-y-3">
                      {pendingRequests.map(req => (
                        <div key={req.role.id} className="flex items-center justify-between rounded-[22px] border border-amber-500/20 bg-amber-500/5 p-4 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-card">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center"><span className="text-sm font-medium text-amber-600">{req.doctor.full_name.split(" ").map(n => n[0]).join("")}</span></div>
                            <div><p className="font-medium text-foreground">{req.doctor.full_name}</p><p className="text-xs text-muted-foreground">{req.doctor.email}</p><Badge variant="secondary" className="mt-1 text-xs">{req.doctor.specialization}</Badge></div>
                          </div>
                          <div className="flex gap-2">
                            <Button size="sm" className="gap-1 bg-primary hover:bg-primary/90" onClick={() => handleApprove(req)}><CheckCircle className="w-3.5 h-3.5" />Approve</Button>
                            <Button size="sm" variant="outline" className="gap-1 text-destructive border-destructive/30 hover:bg-destructive/10" onClick={() => handleReject(req)}><XCircle className="w-3.5 h-3.5" />Reject</Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Admins Tab */}
            {activeTab === "admins" && (
              <Card className="surface-card">
                <CardHeader><CardTitle className="flex items-center gap-2"><Crown className="w-5 h-5 text-amber-500" />Platform Administrators</CardTitle></CardHeader>
                <CardContent>
                  {adminRoles.length === 0 ? (
                    <p className="text-center py-8 text-muted-foreground">No admin roles found.</p>
                  ) : (
                    <div className="space-y-2">
                      {adminRoles.map(ar => {
                        const doc = doctors.find(d => d.user_id === ar.user_id);
                        return (
                          <div key={ar.id} className="flex items-center justify-between rounded-[22px] border border-border/50 bg-white/70 p-3 transition-all duration-300 hover:border-primary/25 hover:shadow-sm">
                            <div className="flex items-center gap-3">
                              <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center"><Crown className="w-4 h-4 text-amber-500" /></div>
                              <div><p className="font-medium text-foreground">{doc?.full_name || "Admin User"}</p><p className="text-xs text-muted-foreground">{doc?.email || ar.user_id}</p></div>
                            </div>
                            <Badge variant="outline" className="bg-amber-500/20 text-amber-600 border-amber-500/30">Admin</Badge>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  <Button variant="outline" className="w-full mt-4 gap-2" onClick={() => setAdminModalOpen(true)}><Plus className="w-4 h-4" />Promote User to Admin</Button>
                </CardContent>
              </Card>
            )}

            {/* HIL Feedback Tab */}
            {activeTab === "hil" && (<>
              <Card className="surface-card">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2"><Brain className="w-5 h-5 text-primary" />Human-in-the-Loop Tasks</CardTitle>
                    <Button className="gap-2 shadow-glow" onClick={() => setHilModalOpen(true)}><Plus className="w-4 h-4" />Create Task</Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {hilTasks.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <Brain className="w-12 h-12 mx-auto mb-3 opacity-40" />
                      <p>No HIL tasks yet. Create one to start collecting expert feedback.</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {hilTasks.map(t => {
                        const doc = doctors.find(d => d.user_id === t.doctor_id);
                        const pct = t.total_scans > 0 ? Math.round((t.completed_scans / t.total_scans) * 100) : 0;
                        return (
                          <div key={t.id} className="space-y-3 rounded-[24px] border border-border/50 bg-white/70 p-4 transition-all duration-300 hover:-translate-y-0.5 hover:border-primary/25 hover:shadow-card">
                            <div className="flex items-center justify-between">
                              <div>
                                <h4 className="font-semibold text-foreground">{t.title}</h4>
                                <p className="text-sm text-muted-foreground">Assigned to: {doc?.full_name || "Unknown"}</p>
                              </div>
                              <Badge variant="outline" className={cn(
                                t.status === "completed" ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30" :
                                t.status === "reviewed" ? "bg-primary/10 text-primary border-primary/30" :
                                t.status === "in_progress" ? "bg-amber-500/10 text-amber-600 border-amber-500/30" :
                                "bg-muted text-muted-foreground"
                              )}>{t.status.replace("_", " ")}</Badge>
                            </div>
                            <div className="space-y-1">
                              <div className="flex justify-between text-xs text-muted-foreground">
                                <span>{t.completed_scans}/{t.total_scans} labeled</span>
                                <span>{pct}%</span>
                              </div>
                              <Progress value={pct} className="h-2" />
                            </div>
                            <div className="flex gap-2 items-center">
                              <Button variant="outline" size="sm" className="gap-1" onClick={() => openHilReview(t.id)}><Eye className="w-3.5 h-3.5" />Review Reports</Button>
                              {(t.approved_scans ?? 0) >= 5 && t.status !== "reviewed" && (
                                <Button size="sm" className="gap-1" onClick={() => handleRunFinetune(t.id)} disabled={hilFinetuning}>
                                  <Play className="w-3.5 h-3.5" />{hilFinetuning ? "Training..." : "Run Fine-tuning"}
                                </Button>
                              )}
                              {((t.approved_scans ?? 0) > 0) && (t.approved_scans ?? 0) < 5 && t.status !== "reviewed" && (
                                <span className="text-xs text-muted-foreground">Need {5 - (t.approved_scans ?? 0)} more approved to fine-tune</span>
                              )}
                              <Button variant="ghost" size="icon" className="ml-auto text-destructive/60 hover:text-destructive" onClick={() => handleDeleteTask(t.id)} title="Delete task">
                                <Trash2 className="w-4 h-4" />
                              </Button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Clinical Feedback Batches */}
              <Card className="surface-card mt-6">
                <CardHeader>
                  <div className="flex items-center justify-between flex-wrap gap-3">
                    <CardTitle className="flex items-center gap-2">
                      <Upload className="w-5 h-5 text-primary" />
                      Vision Agent HIL Batches
                    </CardTitle>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" className="gap-1.5" onClick={fetchHilFeedbackBatches}>
                        <Activity className="w-3.5 h-3.5" /> Refresh
                      </Button>
                      <Button size="sm" className="gap-1.5 shadow-glow" onClick={openCreateFeedbackBatch}>
                        <Plus className="w-3.5 h-3.5" /> New Batch
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {hilFeedbackBatches.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Upload className="w-10 h-10 mx-auto mb-3 opacity-40" />
                      <p>No clinical feedback batches yet.</p>
                      <p className="text-xs mt-1 opacity-70">Select approved feedback below, then create a batch for Vision Agent fine-tuning.</p>
                    </div>
                  ) : (
                    <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
                      <div className="space-y-2">
                        {hilFeedbackBatches.map(batch => {
                          const itemCount = getBatchItemCount(batch.id);
                          const latestJob = getBatchLatestJob(batch.id);
                          const selected = selectedFeedbackBatchId === batch.id;
                          return (
                            <button
                              key={batch.id}
                              type="button"
                              onClick={() => setSelectedFeedbackBatchId(batch.id)}
                              className={cn(
                                "w-full rounded-[18px] border px-3 py-3 text-left transition-all",
                                selected ? "border-primary/60 bg-primary/8 ring-2 ring-primary/20" : "border-border/50 bg-white/70 hover:border-primary/30"
                              )}
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div>
                                  <p className="text-sm font-semibold text-foreground">{batch.title}</p>
                                  <p className="text-xs text-muted-foreground">{itemCount} example{itemCount === 1 ? "" : "s"}</p>
                                </div>
                                <Badge variant="outline" className="text-[10px]">{latestJob?.status || batch.status}</Badge>
                              </div>
                            </button>
                          );
                        })}
                      </div>

                      <div className="rounded-[22px] border border-border/50 bg-white/70 p-4">
                        {selectedFeedbackBatch ? (
                          <div className="space-y-4">
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <h4 className="font-semibold text-foreground">{selectedFeedbackBatch.title}</h4>
                                <p className="text-xs text-muted-foreground">
                                  {selectedFeedbackBatch.instructions || "No special instructions."}
                                </p>
                              </div>
                              <div className="flex flex-wrap justify-end gap-2">
                                <Button variant="outline" size="sm" onClick={() => openEditFeedbackBatch(selectedFeedbackBatch)}>
                                  <Pencil className="w-3.5 h-3.5 mr-1" /> Edit
                                </Button>
                                <Button
                                  size="sm"
                                  className="gap-1"
                                  disabled={queueingFeedbackBatchId === selectedFeedbackBatch.id || selectedFeedbackBatchItems.length < 3}
                                  onClick={() => handleQueueFeedbackBatch(selectedFeedbackBatch.id)}
                                >
                                  <Play className="w-3.5 h-3.5" />
                                  {queueingFeedbackBatchId === selectedFeedbackBatch.id ? "Queueing..." : "Send to Vision Training"}
                                </Button>
                                <Button variant="ghost" size="icon" className="text-destructive/70 hover:text-destructive" onClick={() => handleDeleteFeedbackBatch(selectedFeedbackBatch.id)}>
                                  <Trash2 className="w-4 h-4" />
                                </Button>
                              </div>
                            </div>

                            {selectedFeedbackBatchItems.length === 0 ? (
                              <div className="rounded-[18px] border border-dashed border-border/70 p-6 text-center text-sm text-muted-foreground">
                                No examples in this batch yet.
                              </div>
                            ) : (
                              <div className="space-y-3">
                                {selectedFeedbackBatchItems.map((item, index) => {
                                  const isLegacyHilReport = item.source_type === "hil_report" && !!item.hil_report_id;
                                  const fb = item.feedback_id ? getFeedbackById(item.feedback_id) : undefined;
                                  const legacyReport = item.hil_report_id ? getLegacyHilReportById(item.hil_report_id) : undefined;
                                  const legacyScan = item.hil_scan_id ? getLegacyHilScanById(item.hil_scan_id) : undefined;
                                  if (!fb && !legacyReport) return null;
                                  const previewImageUrl = fb ? getPreviewImageUrl(fb.scan) : legacyScan?.image_url || null;
                                  return (
                                    <div key={item.id} className="rounded-[18px] border border-border/50 bg-white p-3">
                                      <div className="flex items-start justify-between gap-3">
                                        <div className="flex gap-3">
                                          <div className="text-xs font-semibold text-muted-foreground pt-1">#{index + 1}</div>
                                          {previewImageUrl && <img src={previewImageUrl} alt="Scan" className="h-20 w-24 rounded-[12px] border bg-black/5 object-contain" />}
                                          <div className="space-y-1">
                                            {fb ? (
                                              <>
                                                <p className="text-sm font-semibold text-foreground">{fb.scan?.scan_type?.toUpperCase() || "SCAN"} · Anonymous patient {fb.scan?.patient_id?.slice(0, 8) || "unknown"}</p>
                                                <p className="text-xs text-muted-foreground">
                                                  Doctor: {fb.doctor?.full_name || fb.doctor?.email || "Unknown"} · Patient: {fb.patient?.age ? `${fb.patient.age}y` : "age N/A"} {fb.patient?.gender || ""}
                                                </p>
                                                <p className="line-clamp-2 text-xs text-foreground/80">{fb.edited_impression || fb.original_impression || "No impression provided."}</p>
                                              </>
                                            ) : (
                                              <>
                                                <p className="text-sm font-semibold text-foreground">Legacy HIL Report · Task {legacyReport?.task_id.slice(0, 8) || "unknown"}</p>
                                                <p className="text-xs text-muted-foreground">
                                                  Doctor: {doctors.find((doctor) => doctor.user_id === legacyReport?.doctor_id)?.full_name || "Unknown"} · Scan {legacyScan?.scan_order !== undefined ? `#${legacyScan.scan_order + 1}` : "unknown"}
                                                </p>
                                                <p className="line-clamp-2 text-xs text-foreground/80">{legacyReport?.impression || legacyReport?.findings || "No report text provided."}</p>
                                              </>
                                            )}
                                          </div>
                                        </div>
                                        <div className="flex flex-col gap-2 min-w-44">
                                          <Select value={item.batch_id} onValueChange={(batchId) => handleMoveBatchItem(item.id, batchId)}>
                                            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                                            <SelectContent>
                                              {hilFeedbackBatches.map(batch => <SelectItem key={batch.id} value={batch.id}>{batch.title}</SelectItem>)}
                                            </SelectContent>
                                          </Select>
                                          <div className="flex justify-end gap-1">
                                            <Button variant="outline" size="sm" onClick={() => handleRemoveBatchItem(item.id)}>Remove</Button>
                                            {fb && (
                                              <Button variant="ghost" size="icon" className="text-destructive/70 hover:text-destructive" onClick={() => handleDeleteFeedbackFromPool(fb.id, item.id)}>
                                                <Trash2 className="w-4 h-4" />
                                              </Button>
                                            )}
                                          </div>
                                        </div>
                                      </div>
                                      {isLegacyHilReport && legacyReport && (
                                        <div className="mt-3 rounded-[12px] border border-border/50 bg-muted/30 p-2 text-xs text-foreground/80">
                                          <p><span className="font-semibold text-muted-foreground">Findings:</span> {legacyReport.findings || "None"}</p>
                                          <p className="mt-1"><span className="font-semibold text-muted-foreground">Impression:</span> {legacyReport.impression || "None"}</p>
                                        </div>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        ) : (
                          <p className="py-8 text-center text-sm text-muted-foreground">Select a batch to inspect examples.</p>
                        )}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Pending Clinical Approvals */}
              <Card className="surface-card mt-6">
                <CardHeader>
                  <div className="flex items-center justify-between flex-wrap gap-3">
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-emerald-500" />
                      Available Clinical Reports for HIL
                      {availableClinicalFeedback.length > 0 && (
                        <Badge className="ml-1 bg-emerald-500/20 text-emerald-700 border-emerald-500/30 text-xs">
                          {availableClinicalFeedback.length} available
                        </Badge>
                      )}
                    </CardTitle>
                    <div className="flex items-center gap-2">
                      {availableClinicalFeedback.length > 0 && (
                        <>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={toggleSelectAll}
                            className="gap-1.5 text-xs"
                          >
                            {selectedFeedbackIds.size === availableClinicalFeedback.length ? (
                              <XCircle className="w-3.5 h-3.5" />
                            ) : (
                              <CheckCircle className="w-3.5 h-3.5" />
                            )}
                            {selectedFeedbackIds.size === availableClinicalFeedback.length ? "Deselect All" : "Select All"}
                          </Button>
                          {selectedFeedbackIds.size > 0 && (
                            <>
                              <Button
                                size="sm"
                                onClick={handleAddSelectedToFeedbackBatch}
                                disabled={saving || !selectedFeedbackBatchId}
                                className="gap-1.5 shadow-glow text-xs"
                              >
                                <Plus className="w-3.5 h-3.5" />
                                Add {selectedFeedbackIds.size} to Batch
                              </Button>
                              <Button
                                size="sm"
                                onClick={handleBatchApproveClinicalHIL}
                                disabled={batchApproving}
                                variant="outline"
                                className="gap-1.5 text-xs"
                              >
                                <Zap className="w-3.5 h-3.5" />
                                {batchApproving ? "Approving..." : "Mark HIL Approved"}
                              </Button>
                            </>
                          )}
                        </>
                      )}
                      <Button variant="ghost" size="sm" onClick={fetchClinicalFeedback} className="gap-1 text-muted-foreground hover:text-foreground">
                        <Activity className="w-3.5 h-3.5" /> Refresh
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {availableClinicalFeedback.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <CheckCircle className="w-10 h-10 mx-auto mb-3 opacity-40 text-emerald-500" />
                      <p>No available clinical reports to batch.</p>
                      <p className="text-xs mt-1 opacity-60">Reports appear here when a doctor clicks "Approve as Final" and are removed after assignment.</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {availableClinicalFeedback.map(fb => {
                        const previewImageUrl = getPreviewImageUrl(fb.scan);
                        const isSelected = selectedFeedbackIds.has(fb.id);
                        return (
                          <div
                            key={fb.id}
                            onClick={() => toggleFeedbackSelection(fb.id)}
                            className={cn(
                              "space-y-3 rounded-[24px] border p-4 transition-all duration-200 cursor-pointer select-none",
                              isSelected
                                ? "border-emerald-500/60 bg-emerald-50/60 shadow-md ring-2 ring-emerald-500/20"
                                : "border-border/50 bg-white/70 hover:border-emerald-500/30 hover:shadow-card"
                            )}
                          >
                            <div className="flex items-start justify-between gap-3">
                              {/* Checkbox indicator */}
                              <div className={cn(
                                "mt-0.5 flex-shrink-0 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all",
                                isSelected ? "border-emerald-500 bg-emerald-500" : "border-border/60 bg-white"
                              )}>
                                {isSelected && <CheckCircle className="w-3.5 h-3.5 text-white" />}
                              </div>
                              <div className="flex-1">
                                <h4 className="font-semibold text-foreground">Scan ID: {fb.scan_id?.slice(0, 8)}...</h4>
                                <p className="text-sm text-muted-foreground">
                                  Submitted by: {fb.doctor?.full_name || fb.doctor?.email || "Unknown Doctor"}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  Anonymous patient: {fb.scan?.patient_id?.slice(0, 8) || "unknown"} · {fb.patient?.age ? `${fb.patient.age}y` : "age N/A"} {fb.patient?.gender || ""} · {fb.scan?.scan_type?.toUpperCase() || "SCAN"}
                                </p>
                              </div>
                              <Badge variant="outline" className={cn(
                                "text-xs flex-shrink-0",
                                isSelected
                                  ? "bg-emerald-500/20 text-emerald-700 border-emerald-500/40"
                                  : "bg-emerald-500/10 text-emerald-600 border-emerald-500/30"
                              )}>
                                {isSelected ? "✓ Selected" : "Pending Approval"}
                              </Badge>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                              {previewImageUrl ? (
                                <div onClick={e => e.stopPropagation()}>
                                  <img
                                    src={previewImageUrl}
                                    alt="Scan"
                                    className="max-h-[200px] w-full rounded-[16px] border bg-black/5 object-contain"
                                  />
                                </div>
                              ) : (
                                <div className="flex min-h-[120px] items-center justify-center rounded-[16px] border border-dashed border-border/70 bg-muted/30 text-sm text-muted-foreground">
                                  Scan preview unavailable
                                </div>
                              )}
                              <div className="space-y-2 text-sm overflow-y-auto max-h-[200px] pr-2">
                                <div>
                                  <span className="font-semibold text-xs uppercase text-muted-foreground">Findings:</span>
                                  <p className="mt-1">{fb.edited_findings || fb.original_findings}</p>
                                </div>
                                <div>
                                  <span className="font-semibold text-xs uppercase text-muted-foreground">Impression:</span>
                                  <p className="mt-1">{fb.edited_impression || fb.original_impression}</p>
                                </div>
                                {(fb.original_findings || fb.original_impression) && (
                                  <div className="rounded-[12px] border border-border/50 bg-muted/30 p-2">
                                    <span className="font-semibold text-xs uppercase text-muted-foreground">Original report:</span>
                                    <p className="mt-1 text-xs text-muted-foreground">{[fb.original_findings, fb.original_impression].filter(Boolean).join(" ")}</p>
                                  </div>
                                )}
                              </div>
                            </div>

                            <div className="flex justify-end gap-2 mt-3 pt-3 border-t border-border/50" onClick={e => e.stopPropagation()}>
                              <Button
                                onClick={async () => {
                                  if (!selectedFeedbackBatchId) return;
                                  setSaving(true);
                                  try {
                                    await addFeedbackToBatch(selectedFeedbackBatchId, [fb.id]);
                                    toast.success("Example added to batch.");
                                    await fetchHilFeedbackBatches();
                                    await fetchClinicalFeedback();
                                  } catch (e: unknown) {
                                    toast.error(getErrorMessage(e) || "Failed to add example.");
                                  } finally {
                                    setSaving(false);
                                  }
                                }}
                                size="sm"
                                disabled={!selectedFeedbackBatchId}
                                className="gap-2"
                              >
                                <Plus className="w-3.5 h-3.5" />
                                Add to Selected Batch
                              </Button>
                              <Button
                                onClick={() => handleDeleteFeedbackFromPool(fb.id)}
                                size="sm"
                                variant="outline"
                                className="gap-2 text-destructive border-destructive/30 hover:bg-destructive/10"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                                Delete from Pool
                              </Button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <Card className="surface-card">
              <CardHeader><CardTitle className="flex items-center gap-2"><Server className="w-5 h-5 text-primary" />Server Health</CardTitle></CardHeader>
              <CardContent className="space-y-5">
                {healthError ? (
                  <div className="text-center py-4"><WifiOff className="w-10 h-10 mx-auto mb-3 text-destructive/60" /><p className="text-sm text-muted-foreground">FastAPI backend is offline</p></div>
                ) : health ? (
                  <>
                    <div className="flex items-center justify-between"><span className="text-sm text-muted-foreground">Status</span><Badge variant="outline" className="gap-1 bg-primary/20 text-primary border-primary/30"><Wifi className="w-3 h-3" />Online</Badge></div>
                    <div className="flex items-center justify-between"><span className="text-sm text-muted-foreground flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" />Uptime</span><span className="text-sm font-medium text-foreground">{formatUptime(health.uptime_seconds)}</span></div>
                    <MetricBar icon={Cpu} label="CPU" value={health.cpu_percent} max={100} unit="%" />
                    <MetricBar icon={HardDrive} label="RAM" value={health.memory_used_mb} max={health.memory_total_mb} unit="MB" />
                    {health.gpu_available && health.gpu_memory_total_mb && (
                      <div className="pt-2 border-t border-border/50"><p className="text-xs text-muted-foreground mb-2 flex items-center gap-1.5"><Zap className="w-3.5 h-3.5" />{health.gpu_name}</p><MetricBar icon={Zap} label="VRAM" value={health.gpu_memory_used_mb??0} max={health.gpu_memory_total_mb} unit="MB" hideIcon /></div>
                    )}
                  </>
                ) : (
                  <div className="flex items-center justify-center py-6"><Activity className="w-5 h-5 animate-spin text-primary mr-2" /><span className="text-sm text-muted-foreground">Connecting...</span></div>
                )}
              </CardContent>
            </Card>

            <Card className="surface-card">
              <CardHeader><CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5 text-primary" />Platform Summary</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  {[{v:doctors.length,l:"Doctors"},{v:patientCount,l:"Patients"},{v:reportCount,l:"Reports"},{v:activeDoctors,l:"Active"}].map(s => (
                    <div key={s.l} className="rounded-[22px] border border-border/50 bg-white/70 p-4 text-center"><p className="text-2xl font-bold text-primary">{s.v}</p><p className="text-xs text-muted-foreground">{s.l}</p></div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Add/Edit Doctor Modal */}
      <Dialog open={modalOpen} onOpenChange={setModalOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader><DialogTitle>{editingDoctor ? "Edit Doctor" : "Add New Doctor"}</DialogTitle></DialogHeader>
          <div className="space-y-4 py-2">
            <div><label className="text-sm font-medium text-foreground mb-1.5 block">Full Name</label><Input value={formName} onChange={e => setFormName(e.target.value)} placeholder="Dr. Jane Doe" /></div>
            <div><label className="text-sm font-medium text-foreground mb-1.5 block">Email</label><Input value={formEmail} onChange={e => setFormEmail(e.target.value)} placeholder="jane.doe@hospital.com" type="email" /></div>
            <div><label className="text-sm font-medium text-foreground mb-1.5 block">Specialization</label><Select value={formSpec} onValueChange={setFormSpec}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{["Radiology","Cardiology","Pulmonology","Oncology","General Medicine"].map(s => <SelectItem key={s} value={s}>{s}</SelectItem>)}</SelectContent></Select></div>
            {editingDoctor && <div><label className="text-sm font-medium text-foreground mb-1.5 block">Status</label><Select value={formStatus} onValueChange={setFormStatus}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="active">Active</SelectItem><SelectItem value="suspended">Suspended</SelectItem></SelectContent></Select></div>}
          </div>
          <DialogFooter><Button variant="outline" onClick={() => setModalOpen(false)}>Cancel</Button><Button onClick={handleSave} disabled={saving} className="shadow-glow">{saving?"Saving...":editingDoctor?"Update":"Add Doctor"}</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Add Admin Modal */}
      <Dialog open={adminModalOpen} onOpenChange={setAdminModalOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader><DialogTitle>Promote User to Admin</DialogTitle></DialogHeader>
          <div className="space-y-4 py-2">
            <p className="text-sm text-muted-foreground">Enter the email of a registered doctor to promote them to administrator.</p>
            <div><label className="text-sm font-medium text-foreground mb-1.5 block">Doctor Email</label><Input value={adminEmail} onChange={e => setAdminEmail(e.target.value)} placeholder="doctor@hospital.com" type="email" /></div>
          </div>
          <DialogFooter><Button variant="outline" onClick={() => setAdminModalOpen(false)}>Cancel</Button><Button onClick={handleAddAdmin} disabled={saving} className="shadow-glow gap-2"><Crown className="w-4 h-4" />{saving?"Promoting...":"Promote to Admin"}</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create/Edit Clinical HIL Batch Modal */}
      <Dialog open={feedbackBatchModalOpen} onOpenChange={setFeedbackBatchModalOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Upload className="w-5 h-5 text-primary" />
              {editingFeedbackBatch ? "Edit Vision HIL Batch" : "Create Vision HIL Batch"}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">Batch Name</label>
              <Input value={feedbackBatchTitle} onChange={e => setFeedbackBatchTitle(e.target.value)} placeholder="e.g., HIL Vision Batch - May Cases" />
            </div>
            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">Training Notes</label>
              <Textarea value={feedbackBatchInstructions} onChange={e => setFeedbackBatchInstructions(e.target.value)} placeholder="Scope, modality, case mix, or reviewer notes..." className="min-h-[90px]" />
            </div>
            {!editingFeedbackBatch && selectedFeedbackIds.size > 0 && (
              <div className="rounded-[16px] border border-primary/20 bg-primary/5 p-3 text-sm text-muted-foreground">
                {selectedFeedbackIds.size} selected example{selectedFeedbackIds.size === 1 ? "" : "s"} will be added after the batch is created.
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setFeedbackBatchModalOpen(false)}>Cancel</Button>
            <Button onClick={handleSaveFeedbackBatch} disabled={saving} className="shadow-glow">
              {saving ? "Saving..." : editingFeedbackBatch ? "Update Batch" : "Create Batch"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create HIL Task Modal */}
      <Dialog open={hilModalOpen} onOpenChange={(open) => { setHilModalOpen(open); if (!open) { setHilFiles([]); setHilAddToTaskId(""); setHilModalMode("new"); } }}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2"><Brain className="w-5 h-5 text-primary" />Add Scans to HIL Batch</DialogTitle>
          </DialogHeader>

          {/* Mode toggle */}
          <div className="flex rounded-[18px] border border-border/60 overflow-hidden bg-muted/40">
            <button
              type="button"
              onClick={() => setHilModalMode("existing")}
              className={cn(
                "flex-1 py-2.5 text-sm font-medium transition-all duration-200",
                hilModalMode === "existing" ? "bg-primary text-primary-foreground shadow" : "text-muted-foreground hover:text-foreground"
              )}
            >
              Add to Existing Batch
            </button>
            <button
              type="button"
              onClick={() => setHilModalMode("new")}
              className={cn(
                "flex-1 py-2.5 text-sm font-medium transition-all duration-200",
                hilModalMode === "new" ? "bg-primary text-primary-foreground shadow" : "text-muted-foreground hover:text-foreground"
              )}
            >
              Create New Batch
            </button>
          </div>

          {/* Existing batch mode */}
          {hilModalMode === "existing" && (
            <div className="space-y-4 py-1">
              <div>
                <label className="text-sm font-medium mb-2 block text-foreground">Select Existing Batch</label>
                {hilTasks.length === 0 ? (
                  <p className="text-sm text-muted-foreground py-3 text-center border border-dashed rounded-[16px]">No batches yet — create a new one first.</p>
                ) : (
                  <div className="space-y-2 max-h-52 overflow-y-auto pr-1">
                    {hilTasks.map(t => {
                      const doc = doctors.find(d => d.user_id === t.doctor_id);
                      const pct = t.total_scans > 0 ? Math.round((t.completed_scans / t.total_scans) * 100) : 0;
                      const isSelected = hilAddToTaskId === t.id;
                      return (
                        <button
                          key={t.id}
                          type="button"
                          onClick={() => setHilAddToTaskId(t.id)}
                          className={cn(
                            "w-full text-left rounded-[18px] border px-4 py-3 transition-all duration-200",
                            isSelected
                              ? "border-primary/60 bg-primary/8 ring-2 ring-primary/20"
                              : "border-border/50 bg-white/70 hover:border-primary/30"
                          )}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-semibold text-sm text-foreground">{t.title}</span>
                            <Badge variant="outline" className={cn(
                              "text-[10px] px-2",
                              t.status === "completed" ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30" :
                              t.status === "assigned" ? "bg-muted text-muted-foreground" :
                              "bg-amber-500/10 text-amber-600 border-amber-500/30"
                            )}>{t.status}</Badge>
                          </div>
                          <p className="text-xs text-muted-foreground">Assigned to: {doc?.full_name || "Unknown"}</p>
                          <div className="mt-2 flex items-center gap-2">
                            <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                              <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-[10px] text-muted-foreground whitespace-nowrap">{t.completed_scans}/{t.total_scans} labeled</span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
              <div>
                <label className="text-sm font-medium mb-1.5 block">Upload Additional Scans</label>
                <input ref={fileInputRef} type="file" multiple accept="image/*" className="hidden" onChange={e => { if (e.target.files) setHilFiles(Array.from(e.target.files)); }} />
                <Button variant="outline" className="w-full gap-2" onClick={() => fileInputRef.current?.click()}>
                  <Upload className="w-4 h-4" />{hilFiles.length > 0 ? `${hilFiles.length} scans selected` : "Choose files to add"}
                </Button>
              </div>
            </div>
          )}

          {/* New batch mode */}
          {hilModalMode === "new" && (
            <div className="space-y-4 py-1">
              <div><label className="text-sm font-medium mb-1.5 block">Batch Name</label><Input value={hilTitle} onChange={e => setHilTitle(e.target.value)} placeholder="e.g., Batch 3 - Cardiomegaly Review" /></div>
              <div>
                <label className="text-sm font-medium mb-1.5 block">Assign Doctor</label>
                <Select value={hilDoctorId} onValueChange={setHilDoctorId}>
                  <SelectTrigger><SelectValue placeholder="Select a doctor" /></SelectTrigger>
                  <SelectContent>{approvedDoctors.map(d => <SelectItem key={d.user_id} value={d.user_id}>{d.full_name} ({d.email})</SelectItem>)}</SelectContent>
                </Select>
              </div>
              <div><label className="text-sm font-medium mb-1.5 block">Instructions (optional)</label><Textarea value={hilInstructions} onChange={e => setHilInstructions(e.target.value)} placeholder="Specific review instructions for the radiologist..." className="min-h-[72px]" /></div>
              <div>
                <label className="text-sm font-medium mb-1.5 block">Upload X-ray Scans</label>
                <input ref={fileInputRef} type="file" multiple accept="image/*" className="hidden" onChange={e => { if (e.target.files) setHilFiles(Array.from(e.target.files)); }} />
                <Button variant="outline" className="w-full gap-2" onClick={() => fileInputRef.current?.click()}>
                  <Upload className="w-4 h-4" />{hilFiles.length > 0 ? `${hilFiles.length} scans selected` : "Choose files"}
                </Button>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setHilModalOpen(false)}>Cancel</Button>
            {hilModalMode === "existing" ? (
              <Button
                onClick={handleAddScansToTask}
                disabled={saving || !hilAddToTaskId || hilFiles.length === 0}
                className="shadow-glow gap-2"
              >
                <Plus className="w-4 h-4" />{saving ? "Adding..." : `Add ${hilFiles.length > 0 ? hilFiles.length + " " : ""}Scan${hilFiles.length !== 1 ? "s" : ""} to Batch`}
              </Button>
            ) : (
              <Button onClick={handleCreateHilTask} disabled={saving} className="shadow-glow gap-2">
                <Brain className="w-4 h-4" />{saving ? "Creating..." : "Create New Batch"}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* HIL Review Modal */}
      <Dialog open={!!hilReviewTaskId} onOpenChange={() => setHilReviewTaskId(null)}>
        <DialogContent className="sm:max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader><DialogTitle>Review Submitted Reports</DialogTitle></DialogHeader>
          <div className="space-y-6">
            {hilReviewScans.map((s, i) => (
              <div key={s.id} className="space-y-3 rounded-[24px] border border-border/60 bg-white/75 p-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Scan {i + 1}</h4>
                  <Badge variant="outline" className={cn(
                    s.status === "approved" ? "bg-emerald-500/10 text-emerald-600" :
                    s.status === "rejected" ? "bg-destructive/10 text-destructive" :
                    s.report ? "bg-amber-500/10 text-amber-600" : "bg-muted text-muted-foreground"
                  )}>{s.report ? s.status : "Not labeled"}</Badge>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <img src={s.image_url} alt={`Scan ${i+1}`} className="max-h-[300px] w-full rounded-[22px] border bg-black/5 object-contain" />
                  {s.report ? (
                    <div className="space-y-2 text-sm">
                      {s.report.indication && <div><strong className="text-muted-foreground">Indication:</strong> <span>{s.report.indication}</span></div>}
                      {s.report.comparison && <div><strong className="text-muted-foreground">Comparison:</strong> <span>{s.report.comparison}</span></div>}
                      <div><strong className="text-muted-foreground">Findings:</strong> <p className="mt-1 text-foreground leading-relaxed">{s.report.findings}</p></div>
                      <div><strong className="text-muted-foreground">Impression:</strong> <p className="mt-1 text-foreground leading-relaxed">{s.report.impression}</p></div>
                      {s.report.status === "submitted" && (
                        <div className="flex gap-2 pt-2">
                          <Button size="sm" className="gap-1" onClick={() => handleApproveReport(s.report!.id, s.id)}><CheckCircle className="w-3.5 h-3.5" />Approve</Button>
                          <Button size="sm" variant="outline" className="gap-1 text-destructive" onClick={() => handleRejectReport(s.report!.id, s.id)}><XCircle className="w-3.5 h-3.5" />Reject</Button>
                        </div>
                      )}
                      {s.report.status === "approved" && (
                        <div className="mt-2 flex flex-wrap items-center gap-2">
                          <Badge className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30"><CheckCircle className="w-3 h-3 mr-1" />Approved</Badge>
                          <Button size="sm" variant="outline" onClick={() => handleAddApprovedHilReportToBatch(s)} disabled={!selectedFeedbackBatchId || saving}>
                            <Plus className="w-3.5 h-3.5 mr-1" /> Add to Vision Pool
                          </Button>
                        </div>
                      )}
                      {s.report.status === "rejected" && (
                        <div className="mt-2 space-y-1">
                          <Badge className="bg-destructive/10 text-destructive border-destructive/30"><XCircle className="w-3 h-3 mr-1" />Rejected</Badge>
                          {s.report.admin_feedback && <p className="text-xs text-muted-foreground">Feedback: {s.report.admin_feedback}</p>}
                        </div>
                      )}
                      {s.report.status === "draft" && (
                        <Badge variant="outline" className="mt-2 text-muted-foreground">Draft (not yet submitted)</Badge>
                      )}
                    </div>
                  ) : <p className="text-muted-foreground text-sm flex items-center">No report submitted yet.</p>}
                </div>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>

      {/* Fine-tune Confirm Dialog */}
      <Dialog open={!!finetuneConfirmTaskId} onOpenChange={() => setFinetuneConfirmTaskId(null)}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Play className="w-5 h-5 text-primary" />
              Start Fine-Tuning?
            </DialogTitle>
          </DialogHeader>
          <div className="py-2 space-y-3">
            <p className="text-sm text-muted-foreground">
              This will start HIL fine-tuning using all <span className="font-semibold text-foreground">approved reports</span> from the batch:
            </p>
            {finetuneConfirmTaskId && (() => {
              const t = hilTasks.find(x => x.id === finetuneConfirmTaskId);
              return t ? (
                <div className="rounded-[18px] border border-primary/20 bg-primary/5 px-4 py-3">
                  <p className="font-semibold text-foreground">{t.title}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{t.approved_scans ?? t.completed_scans} approved scan{(t.approved_scans ?? t.completed_scans) !== 1 ? "s" : ""} · {doctors.find(d => d.user_id === t.doctor_id)?.full_name || "Unknown"}</p>
                </div>
              ) : null;
            })()}
            <p className="text-xs text-muted-foreground">The model will be retrained in the background. You'll see a notification when complete.</p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setFinetuneConfirmTaskId(null)}>Cancel</Button>
            <Button
              onClick={() => finetuneConfirmTaskId && doRunFinetune(finetuneConfirmTaskId)}
              className="gap-2 shadow-glow"
            >
              <Play className="w-4 h-4" /> Yes, Start Training
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Fine-tune Success Dialog */}
      <Dialog open={finetuneSuccessOpen} onOpenChange={setFinetuneSuccessOpen}>
        <DialogContent className="sm:max-w-sm">
          <div className="flex flex-col items-center text-center py-4 space-y-4">
            <div className="w-16 h-16 rounded-full bg-emerald-500/15 flex items-center justify-center animate-in zoom-in duration-300">
              <Sparkles className="w-8 h-8 text-emerald-500" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-foreground">Batch Sent for Training!</h3>
              <p className="text-sm text-muted-foreground mt-1">
                <span className="font-semibold text-foreground">{finetuneSuccessInfo.taskTitle}</span> is now being fine-tuned.
              </p>
            </div>
            <div className="rounded-[18px] border border-emerald-500/30 bg-emerald-500/8 px-6 py-3 w-full">
              <p className="text-3xl font-bold text-emerald-600">{finetuneSuccessInfo.samples}</p>
              <p className="text-xs text-muted-foreground">training samples queued</p>
            </div>
            <p className="text-xs text-muted-foreground">The model retrains in the background. You'll be notified when it's done.</p>
            <Button onClick={() => setFinetuneSuccessOpen(false)} className="w-full gap-2">
              <CheckCircle className="w-4 h-4" /> Got it
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Layout>
  );
};

function StatCard({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: number }) {
  return (<Card className="metric-card"><CardContent className="pt-6"><div className="flex items-center gap-4"><div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/15 text-primary"><Icon className="h-6 w-6" /></div><div><p className="text-2xl font-bold text-foreground">{value}</p><p className="text-sm text-muted-foreground">{label}</p></div></div></CardContent></Card>);
}

function MetricBar({ icon: Icon, label, value, max, unit, hideIcon }: { icon: LucideIcon; label: string; value: number; max: number; unit: string; hideIcon?: boolean }) {
  const pct = Math.min(100, Math.round((value/max)*100));
  const bar = pct > 85 ? "bg-destructive" : pct > 60 ? "bg-warning" : "bg-primary";
  return (<div><div className="flex items-center justify-between mb-1.5"><span className="text-sm text-muted-foreground flex items-center gap-1.5">{!hideIcon && <Icon className="w-3.5 h-3.5" />}{label}</span><span className="text-xs text-muted-foreground">{value.toLocaleString()}/{max.toLocaleString()} {unit}</span></div><div className="w-full h-2 rounded-full bg-muted overflow-hidden"><div className={cn("h-full rounded-full transition-all duration-500", bar)} style={{width:`${pct}%`}} /></div></div>);
}

export default AdminDashboard;
