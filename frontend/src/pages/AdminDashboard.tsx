import { useState, useEffect, useCallback } from "react";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Users,
  Search,
  Activity,
  Plus,
  UserCog,
  Stethoscope,
  FileText,
  Server,
  Cpu,
  HardDrive,
  Clock,
  Trash2,
  Pencil,
  ShieldCheck,
  ShieldOff,
  Wifi,
  WifiOff,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabaseClient";
import { toast } from "sonner";

// ---------- Types ----------
interface Doctor {
  id: string;
  user_id: string;
  full_name: string;
  email: string;
  specialization: string;
  status: string;
  created_at: string;
}

interface HealthData {
  status: string;
  uptime_seconds: number;
  cpu_percent: number;
  memory_used_mb: number;
  memory_total_mb: number;
  gpu_available: boolean;
  gpu_name?: string;
  gpu_memory_used_mb?: number;
  gpu_memory_total_mb?: number;
}

// ---------- Helpers ----------
function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

// ---------- Component ----------
const AdminDashboard = () => {
  // --- State ---
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [patientCount, setPatientCount] = useState(0);
  const [reportCount, setReportCount] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [health, setHealth] = useState<HealthData | null>(null);
  const [healthError, setHealthError] = useState(false);

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [editingDoctor, setEditingDoctor] = useState<Doctor | null>(null);
  const [formName, setFormName] = useState("");
  const [formEmail, setFormEmail] = useState("");
  const [formSpec, setFormSpec] = useState("Radiology");
  const [formStatus, setFormStatus] = useState("active");
  const [saving, setSaving] = useState(false);

  // --- Data Fetching ---
  const fetchDoctors = useCallback(async () => {
    const { data, error } = await supabase
      .from("doctors")
      .select("*")
      .order("created_at", { ascending: false });
    if (!error && data) setDoctors(data);
  }, []);

  const fetchCounts = useCallback(async () => {
    const [p, r] = await Promise.all([
      supabase.from("patients").select("id", { count: "exact", head: true }),
      supabase.from("reports").select("id", { count: "exact", head: true }),
    ]);
    setPatientCount(p.count ?? 0);
    setReportCount(r.count ?? 0);
  }, []);

  const fetchHealth = useCallback(async () => {
    try {
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
      const res = await fetch(`${API_BASE_URL}/api/health`, {
        signal: AbortSignal.timeout(3000),
        headers: { "ngrok-skip-browser-warning": "true" }
      });
      if (res.ok) {
        setHealth(await res.json());
        setHealthError(false);
      } else {
        setHealthError(true);
      }
    } catch {
      setHealthError(true);
    }
  }, []);

  useEffect(() => {
    fetchDoctors();
    fetchCounts();
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000); // poll every 10s
    return () => clearInterval(interval);
  }, [fetchDoctors, fetchCounts, fetchHealth]);

  // --- CRUD Handlers ---
  const openAdd = () => {
    setEditingDoctor(null);
    setFormName("");
    setFormEmail("");
    setFormSpec("Radiology");
    setFormStatus("active");
    setModalOpen(true);
  };

  const openEdit = (doc: Doctor) => {
    setEditingDoctor(doc);
    setFormName(doc.full_name);
    setFormEmail(doc.email);
    setFormSpec(doc.specialization);
    setFormStatus(doc.status);
    setModalOpen(true);
  };

  const handleSave = async () => {
    if (!formName.trim() || !formEmail.trim()) {
      toast.error("Name and Email are required.");
      return;
    }
    setSaving(true);
    try {
      if (editingDoctor) {
        // UPDATE
        const { error } = await supabase
          .from("doctors")
          .update({
            full_name: formName.trim(),
            email: formEmail.trim(),
            specialization: formSpec,
            status: formStatus,
          })
          .eq("id", editingDoctor.id);
        if (error) throw error;
        toast.success("Doctor updated successfully.");
      } else {
        // INSERT – user_id placeholder (admin manually assigns later or auto via auth)
        const { error } = await supabase.from("doctors").insert({
          full_name: formName.trim(),
          email: formEmail.trim(),
          specialization: formSpec,
          status: formStatus,
          user_id: crypto.randomUUID(), // placeholder
        });
        if (error) throw error;
        toast.success("Doctor added successfully.");
      }
      setModalOpen(false);
      fetchDoctors();
    } catch (e: any) {
      toast.error(e.message || "Operation failed.");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (doc: Doctor) => {
    if (!confirm(`Remove Dr. ${doc.full_name}? This action cannot be undone.`)) return;
    const { error } = await supabase.from("doctors").delete().eq("id", doc.id);
    if (error) {
      toast.error(error.message);
    } else {
      toast.success("Doctor removed.");
      fetchDoctors();
    }
  };

  const toggleStatus = async (doc: Doctor) => {
    const newStatus = doc.status === "active" ? "suspended" : "active";
    const { error } = await supabase.from("doctors").update({ status: newStatus }).eq("id", doc.id);
    if (error) {
      toast.error(error.message);
    } else {
      toast.success(`Doctor ${newStatus === "active" ? "activated" : "suspended"}.`);
      fetchDoctors();
    }
  };

  // --- Filtered doctors ---
  const filtered = doctors.filter((d) => {
    const matchSearch =
      d.full_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      d.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchStatus = statusFilter === "all" || d.status === statusFilter;
    return matchSearch && matchStatus;
  });

  // --- Derived stats ---
  const activeDoctors = doctors.filter((d) => d.status === "active").length;

  // --- Render ---
  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground mb-2 flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
                <ShieldCheck className="w-5 h-5 text-primary" />
              </div>
              Admin Dashboard
            </h1>
            <p className="text-muted-foreground">
              Platform oversight — manage doctors, monitor patients &amp; server health
            </p>
          </div>
          <Button className="gap-2 shadow-glow" onClick={openAdd}>
            <Plus className="w-4 h-4" />
            Add Doctor
          </Button>
        </div>

        {/* ── Stat Cards ────────────────────────── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard icon={Stethoscope} label="Registered Doctors" value={doctors.length} color="primary" />
          <StatCard icon={Users} label="Active Doctors" value={activeDoctors} color="primary" />
          <StatCard icon={UserCog} label="Patient Records" value={patientCount} color="primary" />
          <StatCard icon={FileText} label="Total Reports" value={reportCount} color="primary" />
        </div>

        {/* ── Main Grid ─────────────────────────── */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Doctors Table (2/3) */}
          <div className="lg:col-span-2">
            <Card className="border-border/50 bg-card/50 backdrop-blur">
              <CardHeader>
                <div className="flex flex-col md:flex-row md:items-center gap-4">
                  <CardTitle className="flex items-center gap-2">
                    <Stethoscope className="w-5 h-5 text-primary" />
                    Doctor Registry
                  </CardTitle>
                  <div className="flex-1 flex flex-col md:flex-row gap-2">
                    <div className="relative flex-1">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <Input
                        placeholder="Search doctors..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-9 bg-background/50"
                      />
                    </div>
                    <div className="flex gap-2">
                      {["all", "active", "suspended"].map((s) => (
                        <Button
                          key={s}
                          variant={statusFilter === s ? "default" : "outline"}
                          size="sm"
                          onClick={() => setStatusFilter(s)}
                          className="capitalize"
                        >
                          {s === "all" ? "All" : s}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Doctor</TableHead>
                        <TableHead>Specialization</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Joined</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filtered.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                            {doctors.length === 0
                              ? "No doctors registered yet."
                              : "No matching doctors found."}
                          </TableCell>
                        </TableRow>
                      ) : (
                        filtered.map((doc) => (
                          <TableRow key={doc.id} className="group">
                            <TableCell>
                              <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                                  <span className="text-sm font-medium text-primary">
                                    {doc.full_name
                                      .split(" ")
                                      .map((n) => n[0])
                                      .join("")}
                                  </span>
                                </div>
                                <div>
                                  <p className="font-medium text-foreground">{doc.full_name}</p>
                                  <p className="text-xs text-muted-foreground">{doc.email}</p>
                                </div>
                              </div>
                            </TableCell>
                            <TableCell>
                              <Badge variant="secondary">{doc.specialization}</Badge>
                            </TableCell>
                            <TableCell>
                              <Badge
                                variant="outline"
                                className={cn(
                                  "gap-1",
                                  doc.status === "active"
                                    ? "bg-primary/20 text-primary border-primary/30"
                                    : "bg-destructive/20 text-destructive border-destructive/30"
                                )}
                              >
                                {doc.status === "active" ? (
                                  <ShieldCheck className="w-3 h-3" />
                                ) : (
                                  <ShieldOff className="w-3 h-3" />
                                )}
                                {doc.status === "active" ? "Active" : "Suspended"}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-muted-foreground text-sm">
                              {new Date(doc.created_at).toLocaleDateString()}
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                <Button variant="ghost" size="icon" onClick={() => openEdit(doc)} title="Edit">
                                  <Pencil className="w-4 h-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => toggleStatus(doc)}
                                  title={doc.status === "active" ? "Suspend" : "Activate"}
                                >
                                  {doc.status === "active" ? (
                                    <ShieldOff className="w-4 h-4 text-warning" />
                                  ) : (
                                    <ShieldCheck className="w-4 h-4 text-primary" />
                                  )}
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handleDelete(doc)}
                                  title="Remove"
                                >
                                  <Trash2 className="w-4 h-4 text-destructive" />
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar (1/3) */}
          <div className="space-y-6">
            {/* Server Status */}
            <Card className="border-border/50 bg-card/50 backdrop-blur">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="w-5 h-5 text-primary" />
                  Server Health
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                {healthError ? (
                  <div className="text-center py-4">
                    <WifiOff className="w-10 h-10 mx-auto mb-3 text-destructive/60" />
                    <p className="text-sm text-muted-foreground">FastAPI backend is offline</p>
                    <p className="text-xs text-muted-foreground mt-1">Start it with: <code className="px-1 py-0.5 bg-muted rounded text-xs">python app.py</code></p>
                  </div>
                ) : health ? (
                  <>
                    {/* Status badge */}
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Status</span>
                      <Badge variant="outline" className="gap-1 bg-primary/20 text-primary border-primary/30">
                        <Wifi className="w-3 h-3" />
                        Online
                      </Badge>
                    </div>

                    {/* Uptime */}
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5" /> Uptime
                      </span>
                      <span className="text-sm font-medium text-foreground">{formatUptime(health.uptime_seconds)}</span>
                    </div>

                    {/* CPU */}
                    <MetricBar
                      icon={Cpu}
                      label="CPU"
                      value={health.cpu_percent}
                      max={100}
                      unit="%"
                    />

                    {/* RAM */}
                    <MetricBar
                      icon={HardDrive}
                      label="RAM"
                      value={health.memory_used_mb}
                      max={health.memory_total_mb}
                      unit="MB"
                    />

                    {/* GPU */}
                    {health.gpu_available && health.gpu_memory_total_mb && (
                      <>
                        <div className="pt-2 border-t border-border/50">
                          <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1.5">
                            <Zap className="w-3.5 h-3.5" /> {health.gpu_name}
                          </p>
                          <MetricBar
                            icon={Zap}
                            label="VRAM"
                            value={health.gpu_memory_used_mb ?? 0}
                            max={health.gpu_memory_total_mb}
                            unit="MB"
                            hideIcon
                          />
                        </div>
                      </>
                    )}
                  </>
                ) : (
                  <div className="flex items-center justify-center py-6">
                    <Activity className="w-5 h-5 animate-spin text-primary mr-2" />
                    <span className="text-sm text-muted-foreground">Connecting...</span>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <Card className="border-border/50 bg-card/50 backdrop-blur">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="w-5 h-5 text-primary" />
                  Platform Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-background/50 rounded-lg p-4 border border-border/50 text-center">
                    <p className="text-2xl font-bold text-primary">{doctors.length}</p>
                    <p className="text-xs text-muted-foreground">Doctors</p>
                  </div>
                  <div className="bg-background/50 rounded-lg p-4 border border-border/50 text-center">
                    <p className="text-2xl font-bold text-primary">{patientCount}</p>
                    <p className="text-xs text-muted-foreground">Patients</p>
                  </div>
                  <div className="bg-background/50 rounded-lg p-4 border border-border/50 text-center">
                    <p className="text-2xl font-bold text-primary">{reportCount}</p>
                    <p className="text-xs text-muted-foreground">Reports</p>
                  </div>
                  <div className="bg-background/50 rounded-lg p-4 border border-border/50 text-center">
                    <p className="text-2xl font-bold text-primary">{activeDoctors}</p>
                    <p className="text-xs text-muted-foreground">Active</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* ── Add/Edit Doctor Modal ────────────── */}
      <Dialog open={modalOpen} onOpenChange={setModalOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{editingDoctor ? "Edit Doctor" : "Add New Doctor"}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">Full Name</label>
              <Input value={formName} onChange={(e) => setFormName(e.target.value)} placeholder="Dr. Jane Doe" />
            </div>
            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">Email</label>
              <Input
                value={formEmail}
                onChange={(e) => setFormEmail(e.target.value)}
                placeholder="jane.doe@hospital.com"
                type="email"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">Specialization</label>
              <Select value={formSpec} onValueChange={setFormSpec}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["Radiology", "Cardiology", "Pulmonology", "Oncology", "General Medicine"].map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {editingDoctor && (
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">Status</label>
                <Select value={formStatus} onValueChange={setFormStatus}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="suspended">Suspended</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setModalOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={saving} className="shadow-glow">
              {saving ? "Saving..." : editingDoctor ? "Update" : "Add Doctor"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Layout>
  );
};

// ── Reusable Sub-Components ──────────────────────

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: any;
  label: string;
  value: number;
  color: string;
}) {
  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur">
      <CardContent className="pt-6">
        <div className="flex items-center gap-4">
          <div className={`w-12 h-12 rounded-xl bg-${color}/20 flex items-center justify-center`}>
            <Icon className={`w-6 h-6 text-${color}`} />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{value}</p>
            <p className="text-sm text-muted-foreground">{label}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function MetricBar({
  icon: Icon,
  label,
  value,
  max,
  unit,
  hideIcon,
}: {
  icon: any;
  label: string;
  value: number;
  max: number;
  unit: string;
  hideIcon?: boolean;
}) {
  const pct = Math.min(100, Math.round((value / max) * 100));
  const barColor = pct > 85 ? "bg-destructive" : pct > 60 ? "bg-warning" : "bg-primary";

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-sm text-muted-foreground flex items-center gap-1.5">
          {!hideIcon && <Icon className="w-3.5 h-3.5" />} {label}
        </span>
        <span className="text-xs text-muted-foreground">
          {value.toLocaleString()} / {max.toLocaleString()} {unit}
        </span>
      </div>
      <div className="w-full h-2 rounded-full bg-muted overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-500", barColor)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default AdminDashboard;
