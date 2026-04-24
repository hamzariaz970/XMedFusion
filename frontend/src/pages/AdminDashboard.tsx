import { useState, useEffect, useCallback } from "react";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Users, Search, Activity, Plus, UserCog, Stethoscope, FileText, Server, Cpu, HardDrive, Clock, Trash2, Pencil, ShieldCheck, ShieldOff, Wifi, WifiOff, Zap, CheckCircle, XCircle, Crown } from "lucide-react";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabaseClient";
import { toast } from "sonner";
import { useAuth } from "@/context/AuthContext";

interface Doctor { id: string; user_id: string; full_name: string; email: string; specialization: string; status: string; created_at: string; }
interface UserRole { id: string; user_id: string; role: string; approval_status: string; created_at: string; }
interface PendingRequest { doctor: Doctor; role: UserRole; }
interface HealthData { status: string; uptime_seconds: number; cpu_percent: number; memory_used_mb: number; memory_total_mb: number; gpu_available: boolean; gpu_name?: string; gpu_memory_used_mb?: number; gpu_memory_total_mb?: number; }

function formatUptime(s: number) { const h = Math.floor(s/3600); const m = Math.floor((s%3600)/60); return h > 0 ? `${h}h ${m}m` : `${m}m`; }

const AdminDashboard = () => {
  const { isAdmin } = useAuth();
  const [activeTab, setActiveTab] = useState<"doctors"|"pending"|"admins">("doctors");
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
      const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
      const res = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(3000), headers: { "ngrok-skip-browser-warning": "true" } });
      if (res.ok) { setHealth(await res.json()); setHealthError(false); } else setHealthError(true);
    } catch { setHealthError(true); }
  }, []);

  useEffect(() => {
    fetchDoctors(); fetchPending(); fetchAdmins(); fetchCounts(); fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, [fetchDoctors, fetchPending, fetchAdmins, fetchCounts, fetchHealth]);

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
        const { error } = await supabase.from("doctors").insert({ full_name: formName.trim(), email: formEmail.trim(), specialization: formSpec, status: formStatus, user_id: crypto.randomUUID() });
        if (error) throw error;
        toast.success("Doctor added.");
      }
      setModalOpen(false); fetchDoctors();
    } catch (e: any) { toast.error(e.message); } finally { setSaving(false); }
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
    } catch (e: any) { toast.error(e.message); } finally { setSaving(false); }
  };

  const filtered = doctors.filter(d => {
    const ms = d.full_name.toLowerCase().includes(searchTerm.toLowerCase()) || d.email.toLowerCase().includes(searchTerm.toLowerCase());
    const mf = statusFilter === "all" || d.status === statusFilter;
    return ms && mf;
  });
  const activeDoctors = doctors.filter(d => d.status === "active").length;

  const tabs = [
    { id: "doctors" as const, label: "Doctor Registry", icon: Stethoscope },
    { id: "pending" as const, label: `Pending Requests (${pendingRequests.length})`, icon: Clock },
    { id: "admins" as const, label: "Admin Management", icon: Crown },
  ];

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground mb-2 flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center"><ShieldCheck className="w-5 h-5 text-primary" /></div>
              Admin Dashboard
            </h1>
            <p className="text-muted-foreground">Platform oversight — manage doctors, approve requests &amp; monitor health</p>
          </div>
          <div className="flex gap-2">
            <Button className="gap-2 shadow-glow" onClick={openAdd}><Plus className="w-4 h-4" />Add Doctor</Button>
            <Button variant="outline" className="gap-2" onClick={() => setAdminModalOpen(true)}><Crown className="w-4 h-4" />Add Admin</Button>
          </div>
        </div>

        {/* Stat Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          <StatCard icon={Stethoscope} label="Registered Doctors" value={doctors.length} />
          <StatCard icon={Users} label="Active Doctors" value={activeDoctors} />
          <StatCard icon={Clock} label="Pending Requests" value={pendingRequests.length} />
          <StatCard icon={UserCog} label="Patient Records" value={patientCount} />
          <StatCard icon={FileText} label="Total Reports" value={reportCount} />
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b border-border/50 pb-2">
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
              <Card className="border-border/50 bg-card/50 backdrop-blur">
                <CardHeader>
                  <div className="flex flex-col md:flex-row md:items-center gap-4">
                    <CardTitle className="flex items-center gap-2"><Stethoscope className="w-5 h-5 text-primary" />Doctor Registry</CardTitle>
                    <div className="flex-1 flex flex-col md:flex-row gap-2">
                      <div className="relative flex-1"><Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" /><Input placeholder="Search doctors..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} className="pl-9 bg-background/50" /></div>
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
              <Card className="border-border/50 bg-card/50 backdrop-blur">
                <CardHeader><CardTitle className="flex items-center gap-2"><Clock className="w-5 h-5 text-amber-500" />Pending Registration Requests</CardTitle></CardHeader>
                <CardContent>
                  {pendingRequests.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground"><CheckCircle className="w-12 h-12 mx-auto mb-3 text-primary/40" /><p>No pending requests. All caught up!</p></div>
                  ) : (
                    <div className="space-y-3">
                      {pendingRequests.map(req => (
                        <div key={req.role.id} className="flex items-center justify-between p-4 rounded-lg border border-amber-500/20 bg-amber-500/5">
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
              <Card className="border-border/50 bg-card/50 backdrop-blur">
                <CardHeader><CardTitle className="flex items-center gap-2"><Crown className="w-5 h-5 text-amber-500" />Platform Administrators</CardTitle></CardHeader>
                <CardContent>
                  {adminRoles.length === 0 ? (
                    <p className="text-center py-8 text-muted-foreground">No admin roles found.</p>
                  ) : (
                    <div className="space-y-2">
                      {adminRoles.map(ar => {
                        const doc = doctors.find(d => d.user_id === ar.user_id);
                        return (
                          <div key={ar.id} className="flex items-center justify-between p-3 rounded-lg border border-border/50 bg-background/50">
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
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <Card className="border-border/50 bg-card/50 backdrop-blur">
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

            <Card className="border-border/50 bg-card/50 backdrop-blur">
              <CardHeader><CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5 text-primary" />Platform Summary</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  {[{v:doctors.length,l:"Doctors"},{v:patientCount,l:"Patients"},{v:reportCount,l:"Reports"},{v:activeDoctors,l:"Active"}].map(s => (
                    <div key={s.l} className="bg-background/50 rounded-lg p-4 border border-border/50 text-center"><p className="text-2xl font-bold text-primary">{s.v}</p><p className="text-xs text-muted-foreground">{s.l}</p></div>
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
    </Layout>
  );
};

function StatCard({ icon: Icon, label, value }: { icon: any; label: string; value: number }) {
  return (<Card className="border-border/50 bg-card/50 backdrop-blur"><CardContent className="pt-6"><div className="flex items-center gap-4"><div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center"><Icon className="w-6 h-6 text-primary" /></div><div><p className="text-2xl font-bold text-foreground">{value}</p><p className="text-sm text-muted-foreground">{label}</p></div></div></CardContent></Card>);
}

function MetricBar({ icon: Icon, label, value, max, unit, hideIcon }: { icon: any; label: string; value: number; max: number; unit: string; hideIcon?: boolean }) {
  const pct = Math.min(100, Math.round((value/max)*100));
  const bar = pct > 85 ? "bg-destructive" : pct > 60 ? "bg-warning" : "bg-primary";
  return (<div><div className="flex items-center justify-between mb-1.5"><span className="text-sm text-muted-foreground flex items-center gap-1.5">{!hideIcon && <Icon className="w-3.5 h-3.5" />}{label}</span><span className="text-xs text-muted-foreground">{value.toLocaleString()}/{max.toLocaleString()} {unit}</span></div><div className="w-full h-2 rounded-full bg-muted overflow-hidden"><div className={cn("h-full rounded-full transition-all duration-500", bar)} style={{width:`${pct}%`}} /></div></div>);
}

export default AdminDashboard;
