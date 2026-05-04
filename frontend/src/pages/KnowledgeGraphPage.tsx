import { Layout } from "@/components/layout/Layout";
import KnowledgeGraph from "@/components/KnowledgeGraph";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Link, useNavigate } from "react-router-dom";
import { Activity, ArrowRight, Database, Network, ShieldCheck, FileSearch } from "lucide-react";
import { useAnalysis } from "@/context/AnalysisContext";
import { Card, CardContent } from "@/components/ui/card";

const graphHighlights = [
  { label: "Evidence Links", value: "Traceable", icon: ShieldCheck },
  { label: "Clinical Entities", value: "Structured", icon: Database },
  { label: "Report Claims", value: "Connected", icon: Activity },
];

const KnowledgeGraphPage = () => {
  const { knowledgeGraphData, report } = useAnalysis();
  const navigate = useNavigate();

  return (
    <Layout>
      <section className="figma-page-shell">
        <div className="space-y-8">
          <div className="figma-workspace-hero flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="animate-fade-in">
              <Badge variant="outline" className="eyebrow mb-4">
                <Network className="h-3.5 w-3.5" />
                Clinical Knowledge Layer
              </Badge>
              <h1 className="max-w-3xl text-3xl font-extrabold tracking-tight text-foreground md:text-5xl">
                Diagnostic <span className="text-primary">Evidence Graph</span>
              </h1>
              <p className="mt-4 max-w-2xl text-muted-foreground">
                Explore anatomy, observations, and diagnoses extracted from the current radiology report as a structured clinical graph.
              </p>
            </div>
            {report && (
              <div className="flex gap-3">
                <Button variant="outline" className="gap-2" onClick={() => navigate("/explainability")}>
                  <FileSearch className="w-4 h-4" />
                  View Interpretability
                </Button>
                <Button className="gap-2 shadow-glow" onClick={() => navigate("/upload")}>
                  <Activity className="w-4 h-4" />
                  Synthesis View
                </Button>
              </div>
            )}
          </div>

          {!knowledgeGraphData ? (
            <Card className="surface-card mx-auto max-w-2xl border-dashed">
              <CardContent className="flex flex-col items-center p-12 text-center">
                <div className="w-16 h-16 rounded-full bg-primary/5 flex items-center justify-center mb-6">
                  <Network className="w-8 h-8 text-primary/30" />
                </div>
                <h3 className="text-xl font-bold mb-2">No Active Graph Data</h3>
                <p className="mb-8 text-muted-foreground max-w-sm">
                  The clinical knowledge graph is generated automatically during report synthesis. Please select a patient and upload a scan to see the results.
                </p>
                <Link to="/patients">
                  <Button className="gap-2">
                    Open Patient Registry
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="grid gap-4 md:grid-cols-3">
                {graphHighlights.map((item, index) => (
                  <div key={item.label} className="metric-card animate-slide-up" style={{ animationDelay: `${index * 90}ms` }}>
                    <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                      <item.icon className="h-5 w-5" />
                    </div>
                    <p className="text-sm text-muted-foreground">{item.label}</p>
                    <p className="text-xl font-bold text-foreground">{item.value}</p>
                  </div>
                ))}
              </div>

              <KnowledgeGraph data={knowledgeGraphData} />
            </>
          )}
        </div>
      </section>
    </Layout>
  );
};

export default KnowledgeGraphPage;
