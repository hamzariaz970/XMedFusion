import { useState, useEffect } from "react";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
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
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  User,
  Search,
  Calendar,
  FileText,
  Activity,
  ChevronRight,
  Plus,
  Clock,
  AlertTriangle,
  CheckCircle,
  Stethoscope,
  Network,
  Image as ImageIcon
} from "lucide-react";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabaseClient";
import { usePatientContext } from "@/context/PatientContext";
import { toast } from "sonner";

interface MedicalScan {
  id: string;
  created_at: string;
  scan_type: string;
  original_image_url: string | null;
  heatmap_image_url: string | null;
  findings: string;
  impression: string;
  recommendation: string | null;
  labels: string[];
  severity: string;
}

const statusConfig = {
  active: { label: "Active", color: "bg-primary/20 text-primary border-primary/30", icon: Activity },
  "follow-up": { label: "Follow-up", color: "bg-warning/20 text-warning border-warning/30", icon: Clock },
  resolved: { label: "Resolved", color: "bg-success/20 text-success border-success/30", icon: CheckCircle },
  critical: { label: "Critical", color: "bg-destructive/20 text-destructive border-destructive/30", icon: AlertTriangle },
};

const severityConfig = {
  mild: { label: "Mild", color: "bg-success/20 text-success" },
  moderate: { label: "Moderate", color: "bg-warning/20 text-warning" },
  severe: { label: "Severe", color: "bg-destructive/20 text-destructive" },
};

const PatientDashboard = () => {
  const { patients, selectedPatient, setSelectedPatient, refreshPatients } = usePatientContext();
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [scans, setScans] = useState<MedicalScan[]>([]);
  const [scansLoading, setScansLoading] = useState(false);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);

  // New Patient Form State
  const [newName, setNewName] = useState("");
  const [newAge, setNewAge] = useState("");
  const [newGender, setNewGender] = useState("");
  const [newConditions, setNewConditions] = useState("");
  const [newNotes, setNewNotes] = useState("");

  const filteredPatients = patients.filter((patient) => {
    const matchesSearch =
      patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      patient.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === "all" || patient.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const stats = {
    total: patients.length,
    active: patients.filter((p) => p.status === "active").length,
    critical: patients.filter((p) => p.status === "critical").length,
    resolved: patients.filter((p) => p.status === "resolved").length,
  };

  useEffect(() => {
    const fetchScans = async () => {
      if (!selectedPatient) {
        setScans([]);
        return;
      }
      setScansLoading(true);
      try {
        const { data, error } = await supabase
          .from('medical_scans')
          .select('*')
          .eq('patient_id', selectedPatient.id)
          .order('created_at', { ascending: false });

        if (error) throw error;
        setScans(data || []);
      } catch (err: any) {
        console.error("Error fetching patient scans:", err);
        toast.error("Failed to load patient history");
      } finally {
        setScansLoading(false);
      }
    };

    fetchScans();
  }, [selectedPatient]);

  const handleAddPatient = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error("Not authenticated");

      const conditionsArray = newConditions
        .split(",")
        .map(c => c.trim())
        .filter(c => c.length > 0);

      const { error } = await supabase.from('patients').insert([{
        user_id: user.id,
        name: newName,
        age: parseInt(newAge),
        gender: newGender,
        conditions: conditionsArray,
        notes: newNotes,
        status: 'active'
      }]);

      if (error) throw error;

      toast.success("Patient added successfully");
      setIsAddDialogOpen(false);
      setNewName("");
      setNewAge("");
      setNewGender("");
      setNewConditions("");
      setNewNotes("");
      await refreshPatients();
    } catch (err: any) {
      console.error("Failed to add patient:", err);
      toast.error(err.message || "Failed to add patient");
    }
  };

  const handleUpdateStatus = async (patientId: string, newStatus: string) => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        toast.error("Not authenticated");
        return;
      }

      const { error } = await supabase.from('patients')
        .update({ status: newStatus })
        .eq('id', patientId);

      if (error) throw error;

      toast.success(`Patient status updated to ${newStatus}`);
      await refreshPatients();
    } catch (err: any) {
      console.error("Failed to update status:", err);
      toast.error("Failed to update status");
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric'
    });
  };

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground mb-2">Patient Dashboard</h1>
            <p className="text-muted-foreground">
              Track and manage patient cases and diagnostic history
            </p>
          </div>

          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button className="gap-2 shadow-glow">
                <Plus className="w-4 h-4" />
                Add New Patient
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
              <DialogHeader>
                <DialogTitle>Add New Patient</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleAddPatient} className="space-y-4 mt-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Full Name</label>
                  <Input required value={newName} onChange={e => setNewName(e.target.value)} placeholder="John Doe" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Age</label>
                    <Input required type="number" min="0" value={newAge} onChange={e => setNewAge(e.target.value)} placeholder="45" />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Gender</label>
                    <Select value={newGender} onValueChange={setNewGender} required>
                      <SelectTrigger>
                        <SelectValue placeholder="Select" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Male">Male</SelectItem>
                        <SelectItem value="Female">Female</SelectItem>
                        <SelectItem value="Other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Existing Conditions (comma separated)</label>
                  <Input value={newConditions} onChange={e => setNewConditions(e.target.value)} placeholder="Hypertension, Asthma" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Notes</label>
                  <Textarea value={newNotes} onChange={e => setNewNotes(e.target.value)} placeholder="Any additional context..." />
                </div>
                <Button type="submit" className="w-full">Save Patient</Button>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center">
                  <User className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-foreground">{stats.total}</p>
                  <p className="text-sm text-muted-foreground">Total Patients</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-foreground">{stats.active}</p>
                  <p className="text-sm text-muted-foreground">Active Cases</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-destructive/20 flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-destructive" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-foreground">{stats.critical}</p>
                  <p className="text-sm text-muted-foreground">Critical</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-success/20 flex items-center justify-center">
                  <CheckCircle className="w-6 h-6 text-success" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-foreground">{stats.resolved}</p>
                  <p className="text-sm text-muted-foreground">Resolved</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Patient List */}
          <div className="lg:col-span-2">
            <Card className="border-border/50 bg-card/50 backdrop-blur h-[800px] flex flex-col">
              <CardHeader>
                <div className="flex flex-col md:flex-row md:items-center gap-4">
                  <CardTitle className="flex items-center gap-2 whitespace-nowrap">
                    <User className="w-5 h-5 text-primary" />
                    Patient Registry
                  </CardTitle>
                  <div className="flex-1 flex flex-col md:flex-row gap-2">
                    <div className="relative flex-1">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <Input
                        placeholder="Search patients..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-9 bg-background/50"
                      />
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {["all", "active", "critical", "follow-up", "resolved"].map((status) => (
                        <Button
                          key={status}
                          variant={statusFilter === status ? "default" : "outline"}
                          size="sm"
                          onClick={() => setStatusFilter(status)}
                          className="capitalize"
                        >
                          {status === "all" ? "All" : status}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="flex-1 overflow-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Patient</TableHead>
                      <TableHead>Age/Gender</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Conditions</TableHead>
                      <TableHead>Last Visit</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredPatients.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                          No patients found. Add a new patient to get started.
                        </TableCell>
                      </TableRow>
                    ) : (
                      filteredPatients.map((patient) => {
                        const statusDetails = statusConfig[patient.status as keyof typeof statusConfig] || statusConfig.active;
                        const StatusIcon = statusDetails.icon;
                        return (
                          <TableRow
                            key={patient.id}
                            className={cn(
                              "cursor-pointer transition-colors",
                              selectedPatient?.id === patient.id && "bg-primary/10"
                            )}
                            onClick={() => setSelectedPatient(patient)}
                          >
                            <TableCell>
                              <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-full bg-primary/20 flex flex-shrink-0 items-center justify-center">
                                  <span className="text-sm font-medium text-primary uppercase">
                                    {patient.name.split(" ").map((n) => n[0]).join("").substring(0, 2)}
                                  </span>
                                </div>
                                <div>
                                  <p className="font-medium text-foreground whitespace-nowrap">{patient.name}</p>
                                  <p className="text-xs text-muted-foreground truncate w-24" title={patient.id}>{patient.id.substring(0, 8)}...</p>
                                </div>
                              </div>
                            </TableCell>
                            <TableCell>
                              <span className="text-muted-foreground whitespace-nowrap">
                                {patient.age} / {patient.gender}
                              </span>
                            </TableCell>
                            <TableCell>
                              <Badge
                                variant="outline"
                                className={cn(
                                  "gap-1 whitespace-nowrap",
                                  statusDetails.color
                                )}
                              >
                                <StatusIcon className="w-3 h-3" />
                                {statusDetails.label}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <div className="flex flex-wrap gap-1 max-w-[150px]">
                                {patient.conditions && patient.conditions.length > 0 ? (
                                  <>
                                    {patient.conditions.slice(0, 1).map((condition) => (
                                      <Badge key={condition} variant="secondary" className="text-[10px] truncate max-w-full">
                                        {condition}
                                      </Badge>
                                    ))}
                                    {patient.conditions.length > 1 && (
                                      <Badge variant="secondary" className="text-[10px]">
                                        +{patient.conditions.length - 1}
                                      </Badge>
                                    )}
                                  </>
                                ) : (
                                  <span className="text-xs text-muted-foreground">None</span>
                                )}
                              </div>
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center gap-1 text-muted-foreground whitespace-nowrap">
                                <Calendar className="w-3 h-3" />
                                <span className="text-xs">{formatDate(patient.updated_at)}</span>
                              </div>
                            </TableCell>
                            <TableCell>
                              <ChevronRight className="w-4 h-4 text-muted-foreground" />
                            </TableCell>
                          </TableRow>
                        );
                      })
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>

          {/* Patient Details & History */}
          <div className="space-y-6">
            {selectedPatient ? (
              <>
                {/* Patient Info Card */}
                <Card className="border-border/50 bg-card/50 backdrop-blur">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="w-5 h-5 text-primary" />
                      Patient Details
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-16 rounded-full bg-primary/20 flex flex-shrink-0 items-center justify-center">
                        <span className="text-xl font-bold text-primary uppercase">
                          {selectedPatient.name.split(" ").map((n) => n[0]).join("").substring(0, 2)}
                        </span>
                      </div>
                      <div className="min-w-0">
                        <h3 className="text-lg font-semibold text-foreground truncate">
                          {selectedPatient.name}
                        </h3>
                        <p className="text-sm text-muted-foreground truncate">
                          {selectedPatient.age} years • {selectedPatient.gender}
                        </p>
                      </div>
                    </div>

                    {selectedPatient.notes && (
                      <div className="pt-2">
                        <p className="text-xs text-muted-foreground">{selectedPatient.notes}</p>
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border/50">
                      <div>
                        <p className="text-xs text-muted-foreground uppercase tracking-wide">Total Scans</p>
                        <p className="text-lg font-semibold text-foreground">{scans.length}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground uppercase tracking-wide">Status</p>
                        <Select
                          value={selectedPatient.status || 'active'}
                          onValueChange={(val) => handleUpdateStatus(selectedPatient.id, val)}
                        >
                          <SelectTrigger className="w-[140px] h-8 mt-1 border-border/50 bg-background/50 text-xs font-medium">
                            <SelectValue placeholder="Select Status" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="active">
                              <span className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-blue-500" /> Active
                              </span>
                            </SelectItem>
                            <SelectItem value="resolved">
                              <span className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-emerald-500" /> Resolved
                              </span>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="pt-4 border-t border-border/50">
                      <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                        Conditions
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {selectedPatient.conditions && selectedPatient.conditions.length > 0 ? (
                          selectedPatient.conditions.map((condition) => (
                            <Badge key={condition} variant="secondary">
                              {condition}
                            </Badge>
                          ))
                        ) : (
                          <span className="text-sm text-muted-foreground">None documented</span>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Diagnostic History */}
                <Card className="border-border/50 bg-card/50 backdrop-blur flex-1">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-primary" />
                      Diagnostic History
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="max-h-[400px] overflow-y-auto">
                    <div className="space-y-4">
                      {scansLoading ? (
                        <div className="text-center text-sm text-muted-foreground py-4">Loading history...</div>
                      ) : scans.length === 0 ? (
                        <div className="text-center text-sm text-muted-foreground py-4">No diagnostic history found.</div>
                      ) : (
                        scans.map((scan, index) => (
                          <div
                            key={scan.id}
                            className={cn(
                              "relative pl-6 pb-4",
                              index !== scans.length - 1 && "border-l-2 border-border/50"
                            )}
                          >
                            <div className="absolute left-0 top-0 w-3 h-3 rounded-full bg-primary -translate-x-[5px]" />
                            <div className="bg-background/50 rounded-lg p-4 border border-border/50 hover:border-primary/30 transition-colors">
                              <div className="flex items-center justify-between mb-3">
                                <span className="text-sm font-semibold text-foreground uppercase tracking-wider">
                                  {scan.scan_type === 'auto' ? 'Medical Scan' : scan.scan_type}
                                </span>
                                <Badge
                                  variant="outline"
                                  className={
                                    severityConfig[scan.severity as keyof typeof severityConfig]?.color || severityConfig.moderate.color
                                  }
                                >
                                  {severityConfig[scan.severity as keyof typeof severityConfig]?.label || 'Moderate'}
                                </Badge>
                              </div>

                              <div className="space-y-3 mb-4">
                                <div>
                                  <h4 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-1">Findings</h4>
                                  <p className="text-sm text-foreground/90 leading-relaxed">
                                    {scan.findings || "No findings recorded."}
                                  </p>
                                </div>
                                {scan.impression && (
                                  <div>
                                    <h4 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-1">Impression</h4>
                                    <p className="text-sm text-foreground/90 leading-relaxed font-medium">
                                      {scan.impression}
                                    </p>
                                  </div>
                                )}
                                {scan.recommendation && (
                                  <div>
                                    <h4 className="text-xs font-semibold uppercase text-primary/80 tracking-wider mb-1">Recommendation</h4>
                                    <p className="text-sm text-foreground/90 leading-relaxed">
                                      {scan.recommendation}
                                    </p>
                                  </div>
                                )}
                              </div>

                              {/* Display Labels if they exist */}
                              {scan.labels && scan.labels.length > 0 && (
                                <div className="flex flex-wrap gap-1 mb-3">
                                  {scan.labels.slice(0, 3).map((label, idx) => (
                                    <span key={idx} className="text-[9px] uppercase font-bold px-2 py-0.5 rounded-full bg-primary/10 text-primary">
                                      {label}
                                    </span>
                                  ))}
                                  {scan.labels.length > 3 && (
                                    <span className="text-[9px] uppercase font-bold px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                                      +{scan.labels.length - 3}
                                    </span>
                                  )}
                                </div>
                              )}

                              <div className="flex flex-col sm:flex-row gap-2 items-start sm:items-center justify-between mt-4 pb-2 border-b border-border/30">
                                <div className="flex flex-wrap items-center gap-3">
                                  {scan.original_image_url.split(',').map((url, imgIndex) => {
                                    const urlValue = url.trim();
                                    if (!urlValue) return null;
                                    return (
                                      <a
                                        key={imgIndex}
                                        href={urlValue}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="flex items-center gap-1 text-xs font-medium text-primary hover:text-primary/80 hover:underline bg-primary/10 px-2 py-1 rounded"
                                      >
                                        <ImageIcon className="w-3.5 h-3.5" /> View Scan {scan.original_image_url.includes(',') ? `#${imgIndex + 1}` : ''}
                                      </a>
                                    );
                                  })}
                                  {scan.heatmap_image_url && (
                                    <a
                                      href={scan.heatmap_image_url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="flex items-center gap-1 text-xs font-medium text-orange-500 hover:text-orange-400 hover:underline bg-orange-500/10 px-2 py-1 rounded"
                                    >
                                      <Activity className="w-3.5 h-3.5" /> Insights
                                    </a>
                                  )}
                                </div>
                                <div className="flex items-center gap-1 text-xs text-muted-foreground ml-auto">
                                  <Calendar className="w-3 h-3" />
                                  <span>{formatDate(scan.created_at)}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card className="border-border/50 bg-card/50 backdrop-blur h-[800px]">
                <CardContent className="flex flex-col justify-center items-center h-full space-y-4">
                  <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                    <User className="w-8 h-8 text-primary opacity-50" />
                  </div>
                  <div className="text-center px-4">
                    <h3 className="text-lg font-medium text-foreground mb-1">No Patient Selected</h3>
                    <p className="text-sm text-muted-foreground max-w-[250px] mx-auto">
                      Select a patient from the registry to view their details, full history, and unlock the analysis tools.
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default PatientDashboard;
