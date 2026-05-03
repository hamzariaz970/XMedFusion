import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";
import { useAuth } from "@/context/AuthContext";
import { usePatientContext, type Patient } from "@/context/PatientContext";
import { supabase } from "@/lib/supabaseClient";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Brain,
  CalendarClock,
  CheckCircle,
  Clock,
  FileText,
  ImagePlus,
  ListChecks,
  Loader2,
  Network,
  Search,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UserRound,
  Users,
} from "lucide-react";

interface DoctorProfile {
  full_name: string;
  specialization: string;
}

interface RecentScan {
  id: string;
  patient_id: string;
  created_at: string;
  scan_type: string | null;
  severity: string | null;
  impression: string | null;
}

interface HilTaskSummary {
  id: string;
  title: string;
  total_scans: number;
  completed_scans: number;
  status: string;
}

const getGreeting = () => {
  const hour = new Date().getHours();
  if (hour < 12) return "Good Morning";
  if (hour < 17) return "Good Afternoon";
  return "Good Evening";
};

const formatRelativeDate = (dateString?: string | null) => {
  if (!dateString) return "No activity yet";
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

const startOfWeek = () => {
  const date = new Date();
  const day = date.getDay();
  date.setDate(date.getDate() - day);
  date.setHours(0, 0, 0, 0);
  return date;
};

const DoctorDashboard = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { patients, selectedPatient, setSelectedPatient, loading: patientsLoading } = usePatientContext();
  const [doctorProfile, setDoctorProfile] = useState<DoctorProfile | null>(null);
  const [recentScans, setRecentScans] = useState<RecentScan[]>([]);
  const [hilTasks, setHilTasks] = useState<HilTaskSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) return;

    const loadDashboard = async () => {
      setLoading(true);
      try {
        const [{ data: doctor }, { data: tasks }] = await Promise.all([
          supabase
            .from("doctors")
            .select("full_name, specialization")
            .eq("user_id", user.id)
            .maybeSingle(),
          supabase
            .from("hil_tasks")
            .select("id, title, total_scans, completed_scans, status")
            .eq("doctor_id", user.id)
            .in("status", ["assigned", "in_progress"])
            .order("created_at", { ascending: false })
            .limit(4),
        ]);

        if (doctor) setDoctorProfile(doctor as DoctorProfile);
        setHilTasks((tasks || []) as HilTaskSummary[]);
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, [user]);

  useEffect(() => {
    const loadRecentScans = async () => {
      if (patients.length === 0) {
        setRecentScans([]);
        return;
      }

      const patientIds = patients.map((patient) => patient.id);
      const { data, error } = await supabase
        .from("medical_scans")
        .select("id, patient_id, created_at, scan_type, severity, impression")
        .in("patient_id", patientIds)
        .order("created_at", { ascending: false })
        .limit(8);

      if (!error && data) {
        setRecentScans(data as RecentScan[]);
      }
    };

    loadRecentScans();
  }, [patients]);

  const doctorName = useMemo(() => {
    const fullName = doctorProfile?.full_name || user?.email?.split("@")[0] || "Doctor";
    return fullName.replace(/^Dr\.?\s+/i, "");
  }, [doctorProfile, user]);

  const patientById = useMemo(() => {
    return patients.reduce<Record<string, Patient>>((acc, patient) => {
      acc[patient.id] = patient;
      return acc;
    }, {});
  }, [patients]);

  const lastPatient = useMemo(() => {
    if (selectedPatient) return selectedPatient;
    if (recentScans[0]) return patientById[recentScans[0].patient_id] || null;
    return [...patients].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())[0] || null;
  }, [patientById, patients, recentScans, selectedPatient]);

  const stats = useMemo(() => {
    const weekStart = startOfWeek();
    const studiesThisWeek = recentScans.filter((scan) => new Date(scan.created_at) >= weekStart).length;
    const criticalCases = patients.filter((patient) => patient.status === "critical").length;
    const activeCases = patients.filter((patient) => patient.status === "active").length;
    const completedHil = hilTasks.reduce((sum, task) => sum + task.completed_scans, 0);
    const totalHil = hilTasks.reduce((sum, task) => sum + task.total_scans, 0);

    return {
      totalPatients: patients.length,
      activeCases,
      criticalCases,
      studiesThisWeek,
      hilProgress: totalHil ? Math.round((completedHil / totalHil) * 100) : 0,
    };
  }, [hilTasks, patients, recentScans]);

  const openPatient = (patient: Patient | null, route = "/patients") => {
    if (patient) setSelectedPatient(patient);
    navigate(route);
  };

  const recentWorklist = recentScans.slice(0, 5);
  const isBusy = loading || patientsLoading;

  return (
    <Layout>
      <div className="figma-page-shell space-y-8">
        <section className="figma-workspace-hero">
          <div>
            <Badge variant="outline" className="eyebrow mb-4">
              <ShieldCheck className="h-3.5 w-3.5" />
              Radiologist Workspace
            </Badge>
            <h1 className="max-w-4xl text-3xl font-extrabold tracking-tight text-foreground md:text-5xl">
              {getGreeting()}, <span className="text-primary">Dr. {doctorName}</span>
            </h1>
            <p className="mt-4 max-w-2xl text-muted-foreground">
              Welcome back. Access your patient registry, start new diagnostic reports, and review active clinical evidence.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <Button className="gap-2 shadow-glow" onClick={() => openPatient(lastPatient, lastPatient ? "/upload" : "/patients")}>
                <ImagePlus className="h-4 w-4" />
                {lastPatient ? `Continue ${lastPatient.name}` : "Select patient"}
              </Button>
              <Button variant="outline" className="gap-2" onClick={() => navigate("/patients")}>
                <Search className="h-4 w-4" />
                Open patient registry
              </Button>
            </div>
          </div>
        </section>

        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          <DashboardMetric icon={Users} label="Patient Panel" value={stats.totalPatients} detail={`${stats.activeCases} active cases`} />
          <DashboardMetric icon={FileText} label="Studies This Week" value={stats.studiesThisWeek} detail="Recent generated reports" />
          <DashboardMetric icon={AlertTriangle} label="Critical Cases" value={stats.criticalCases} detail="Need close follow-up" tone="destructive" />
        </section>

        <section className="grid gap-6 xl:grid-cols-[0.8fr_1.2fr_0.8fr]">
          <Card className="surface-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UserRound className="h-5 w-5 text-primary" />
                Last Patient
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              {lastPatient ? (
                <>
                  <div className="rounded-[24px] border border-primary/10 bg-primary/5 p-5">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-2xl font-extrabold text-foreground">{lastPatient.name}</p>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {lastPatient.age} years / {lastPatient.gender}
                        </p>
                      </div>
                      <Badge variant="outline" className="capitalize">
                        {lastPatient.status}
                      </Badge>
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                      {(lastPatient.conditions || []).slice(0, 3).map((condition) => (
                        <span key={condition} className="medical-chip">
                          {condition}
                        </span>
                      ))}
                      {lastPatient.conditions?.length === 0 && <span className="text-sm text-muted-foreground">No conditions documented</span>}
                    </div>
                  </div>
                  <div className="grid gap-3">
                    <Button className="justify-between" onClick={() => openPatient(lastPatient, "/upload")}>
                      Start report
                      <ArrowRight className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" className="justify-between" onClick={() => openPatient(lastPatient, "/patients")}>
                      Open history
                      <ArrowRight className="h-4 w-4" />
                    </Button>
                  </div>
                </>
              ) : (
                <EmptyState
                  icon={Users}
                  title="No patient selected"
                  copy="Create or select a patient to start a radiology report workflow."
                  action="Open registry"
                  onAction={() => navigate("/patients")}
                />
              )}
            </CardContent>
          </Card>

          <Card className="surface-card">
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <CardTitle className="flex items-center gap-2">
                  <CalendarClock className="h-5 w-5 text-primary" />
                  Recent Imaging Worklist
                </CardTitle>
                <Badge variant="outline">{recentWorklist.length} studies</Badge>
              </div>
            </CardHeader>
            <CardContent>
              {isBusy ? (
                <div className="flex items-center justify-center py-12 text-muted-foreground">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Loading worklist...
                </div>
              ) : recentWorklist.length > 0 ? (
                <div className="space-y-3">
                  {recentWorklist.map((scan) => {
                    const patient = patientById[scan.patient_id];
                    return (
                      <button
                        key={scan.id}
                        type="button"
                        className="w-full rounded-[24px] border border-border/60 bg-white/70 p-4 text-left transition-all duration-300 hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-card"
                        onClick={() => openPatient(patient || null, "/patients")}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <p className="font-bold text-foreground">{patient?.name || "Unknown patient"}</p>
                            <p className="mt-1 text-sm text-muted-foreground">
                              {(scan.scan_type || "Medical scan").toUpperCase()} / {formatRelativeDate(scan.created_at)}
                            </p>
                          </div>
                          <Badge variant="outline" className="capitalize">
                            {scan.severity || "review"}
                          </Badge>
                        </div>
                        <p className="mt-3 line-clamp-2 text-sm leading-6 text-muted-foreground">
                          {scan.impression || "No impression recorded yet."}
                        </p>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <EmptyState
                  icon={FileText}
                  title="No recent studies"
                  copy="Generated reports and uploaded imaging studies will appear here."
                  action="Upload scan"
                  onAction={() => openPatient(lastPatient, lastPatient ? "/upload" : "/patients")}
                />
              )}
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card className="surface-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ListChecks className="h-5 w-5 text-primary" />
                  Clinical Tools
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <PriorityItem icon={AlertTriangle} label="Critical patients" value={stats.criticalCases} active={stats.criticalCases > 0} />
                <PriorityItem icon={Network} label="Evidence Graph" value="Active" />
                <Button 
                  variant="outline" 
                  className="w-full justify-between" 
                  onClick={() => navigate("/knowledge-graph")}
                  disabled={!selectedPatient}
                >
                  Open knowledge graph
                  <ArrowRight className="h-4 w-4" />
                </Button>
                <Button 
                  variant="outline" 
                  className="w-full justify-between" 
                  onClick={() => navigate("/explainability")}
                  disabled={!selectedPatient}
                >
                  View explainability
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </Layout>
  );
};

function DashboardMetric({
  icon: Icon,
  label,
  value,
  detail,
  tone = "primary",
}: {
  icon: any;
  label: string;
  value: number | string;
  detail: string;
  tone?: "primary" | "destructive";
}) {
  return (
    <Card className="metric-card">
      <CardContent className="pt-6">
        <div className="flex items-center gap-4">
          <div className={`flex h-12 w-12 items-center justify-center rounded-2xl ${tone === "destructive" ? "bg-destructive/15 text-destructive" : "bg-primary/15 text-primary"}`}>
            <Icon className="h-6 w-6" />
          </div>
          <div>
            <p className="text-2xl font-extrabold text-foreground">{value}</p>
            <p className="text-sm font-semibold text-foreground">{label}</p>
            <p className="text-xs text-muted-foreground">{detail}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function PriorityItem({
  icon: Icon,
  label,
  value,
  active,
}: {
  icon: any;
  label: string;
  value: number | string;
  active?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-[22px] border border-border/60 bg-white/70 p-3">
      <div className="flex items-center gap-3">
        <div className={`flex h-9 w-9 items-center justify-center rounded-2xl ${active ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground"}`}>
          <Icon className="h-4 w-4" />
        </div>
        <span className="text-sm font-semibold text-foreground">{label}</span>
      </div>
      <span className="text-sm font-bold text-primary">{value}</span>
    </div>
  );
}

function EmptyState({
  icon: Icon,
  title,
  copy,
  action,
  onAction,
}: {
  icon: any;
  title: string;
  copy: string;
  action: string;
  onAction: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-[24px] border border-dashed border-border bg-white/60 p-8 text-center">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
        <Icon className="h-6 w-6" />
      </div>
      <h3 className="font-bold text-foreground">{title}</h3>
      <p className="mt-2 max-w-sm text-sm leading-6 text-muted-foreground">{copy}</p>
      <Button className="mt-5 gap-2" onClick={onAction}>
        {action}
        <ArrowRight className="h-4 w-4" />
      </Button>
    </div>
  );
}

export default DoctorDashboard;
