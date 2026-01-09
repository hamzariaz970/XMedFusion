import { useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  User,
  Search,
  Calendar,
  FileText,
  Activity,
  ChevronRight,
  Plus,
  Filter,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Hardcoded patient data
const patients = [
  {
    id: "P001",
    name: "John Smith",
    age: 45,
    gender: "Male",
    lastVisit: "2024-01-15",
    totalCases: 3,
    status: "active",
    conditions: ["Pneumonia", "Cardiomegaly"],
  },
  {
    id: "P002",
    name: "Sarah Johnson",
    age: 62,
    gender: "Female",
    lastVisit: "2024-01-12",
    totalCases: 5,
    status: "follow-up",
    conditions: ["Lung Nodule", "Pleural Effusion"],
  },
  {
    id: "P003",
    name: "Michael Chen",
    age: 38,
    gender: "Male",
    lastVisit: "2024-01-10",
    totalCases: 2,
    status: "resolved",
    conditions: ["Atelectasis"],
  },
  {
    id: "P004",
    name: "Emily Davis",
    age: 55,
    gender: "Female",
    lastVisit: "2024-01-08",
    totalCases: 4,
    status: "critical",
    conditions: ["Mass Lesion", "Lymphadenopathy"],
  },
  {
    id: "P005",
    name: "Robert Wilson",
    age: 71,
    gender: "Male",
    lastVisit: "2024-01-05",
    totalCases: 6,
    status: "active",
    conditions: ["COPD", "Emphysema", "Bronchiectasis"],
  },
];

// Hardcoded diagnostic history for selected patient
const diagnosticHistory = [
  {
    id: "D001",
    date: "2024-01-15",
    type: "Chest X-ray",
    findings: "Bilateral infiltrates consistent with pneumonia",
    severity: "moderate",
    radiologist: "Dr. Amanda Foster",
  },
  {
    id: "D002",
    date: "2024-01-10",
    type: "CT Scan",
    findings: "Follow-up imaging shows improvement in consolidation",
    severity: "mild",
    radiologist: "Dr. James Lee",
  },
  {
    id: "D003",
    date: "2024-01-02",
    type: "Chest X-ray",
    findings: "Initial presentation with right lower lobe opacity",
    severity: "severe",
    radiologist: "Dr. Amanda Foster",
  },
];

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
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<typeof patients[0] | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("all");

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
          <Button className="gap-2 shadow-glow">
            <Plus className="w-4 h-4" />
            Add New Patient
          </Button>
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
            <Card className="border-border/50 bg-card/50 backdrop-blur">
              <CardHeader>
                <div className="flex flex-col md:flex-row md:items-center gap-4">
                  <CardTitle className="flex items-center gap-2">
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
                    <div className="flex gap-2">
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
              <CardContent>
                <div className="overflow-x-auto">
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
                      {filteredPatients.map((patient) => {
                        const StatusIcon = statusConfig[patient.status as keyof typeof statusConfig].icon;
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
                                <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                                  <span className="text-sm font-medium text-primary">
                                    {patient.name.split(" ").map((n) => n[0]).join("")}
                                  </span>
                                </div>
                                <div>
                                  <p className="font-medium text-foreground">{patient.name}</p>
                                  <p className="text-xs text-muted-foreground">{patient.id}</p>
                                </div>
                              </div>
                            </TableCell>
                            <TableCell>
                              <span className="text-muted-foreground">
                                {patient.age} / {patient.gender}
                              </span>
                            </TableCell>
                            <TableCell>
                              <Badge
                                variant="outline"
                                className={cn(
                                  "gap-1",
                                  statusConfig[patient.status as keyof typeof statusConfig].color
                                )}
                              >
                                <StatusIcon className="w-3 h-3" />
                                {statusConfig[patient.status as keyof typeof statusConfig].label}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <div className="flex flex-wrap gap-1">
                                {patient.conditions.slice(0, 2).map((condition) => (
                                  <Badge key={condition} variant="secondary" className="text-xs">
                                    {condition}
                                  </Badge>
                                ))}
                                {patient.conditions.length > 2 && (
                                  <Badge variant="secondary" className="text-xs">
                                    +{patient.conditions.length - 2}
                                  </Badge>
                                )}
                              </div>
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center gap-1 text-muted-foreground">
                                <Calendar className="w-3 h-3" />
                                <span className="text-sm">{patient.lastVisit}</span>
                              </div>
                            </TableCell>
                            <TableCell>
                              <ChevronRight className="w-4 h-4 text-muted-foreground" />
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
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
                      <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center">
                        <span className="text-xl font-bold text-primary">
                          {selectedPatient.name.split(" ").map((n) => n[0]).join("")}
                        </span>
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-foreground">
                          {selectedPatient.name}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {selectedPatient.id} • {selectedPatient.age} years • {selectedPatient.gender}
                        </p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border/50">
                      <div>
                        <p className="text-xs text-muted-foreground uppercase tracking-wide">Total Cases</p>
                        <p className="text-lg font-semibold text-foreground">{selectedPatient.totalCases}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground uppercase tracking-wide">Status</p>
                        <Badge
                          variant="outline"
                          className={statusConfig[selectedPatient.status as keyof typeof statusConfig].color}
                        >
                          {statusConfig[selectedPatient.status as keyof typeof statusConfig].label}
                        </Badge>
                      </div>
                    </div>
                    <div className="pt-4 border-t border-border/50">
                      <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                        Active Conditions
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {selectedPatient.conditions.map((condition) => (
                          <Badge key={condition} variant="secondary">
                            {condition}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Diagnostic History */}
                <Card className="border-border/50 bg-card/50 backdrop-blur">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-primary" />
                      Diagnostic History
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {diagnosticHistory.map((diagnosis, index) => (
                        <div
                          key={diagnosis.id}
                          className={cn(
                            "relative pl-6 pb-4",
                            index !== diagnosticHistory.length - 1 && "border-l-2 border-border/50"
                          )}
                        >
                          <div className="absolute left-0 top-0 w-3 h-3 rounded-full bg-primary -translate-x-[5px]" />
                          <div className="bg-background/50 rounded-lg p-4 border border-border/50">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-sm font-medium text-foreground">
                                {diagnosis.type}
                              </span>
                              <Badge
                                className={
                                  severityConfig[diagnosis.severity as keyof typeof severityConfig].color
                                }
                              >
                                {severityConfig[diagnosis.severity as keyof typeof severityConfig].label}
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mb-2">{diagnosis.findings}</p>
                            <div className="flex items-center justify-between text-xs text-muted-foreground">
                              <span>{diagnosis.radiologist}</span>
                              <span>{diagnosis.date}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card className="border-border/50 bg-card/50 backdrop-blur">
                <CardContent className="py-12">
                  <div className="text-center text-muted-foreground">
                    <User className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Select a patient to view details</p>
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
