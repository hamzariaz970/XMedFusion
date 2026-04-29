import { Layout } from "@/components/layout/Layout";
import KnowledgeGraph from "@/components/KnowledgeGraph";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { radiologyImages } from "@/assets/radiology";
import { Link } from "react-router-dom";
import { Activity, ArrowRight, Database, Network, ShieldCheck } from "lucide-react";

const graphHighlights = [
  { label: "Evidence Links", value: "Traceable", icon: ShieldCheck },
  { label: "Clinical Entities", value: "Structured", icon: Database },
  { label: "Report Claims", value: "Connected", icon: Activity },
];

const KnowledgeGraphPage = () => {
  return (
    <Layout>
      <section className="figma-page-shell">
        <div className="space-y-8">
          <div className="figma-workspace-hero grid gap-6 lg:grid-cols-[1fr_420px] lg:items-center">
            <div className="animate-fade-in">
              <Badge variant="outline" className="eyebrow mb-4">
                <Network className="h-3.5 w-3.5" />
                Clinical Knowledge Layer
              </Badge>
              <h1 className="max-w-3xl text-3xl font-extrabold tracking-tight text-foreground md:text-5xl">
                Map AI findings into a transparent <span className="text-primary">clinical graph.</span>
              </h1>
              <p className="mt-4 max-w-2xl text-muted-foreground">
                Explore anatomy, observations, diagnoses, and relationships that support evidence-linked radiology reporting.
              </p>
              <Link to="/login" className="mt-8 inline-flex">
                <Button variant="hero" size="lg" className="w-full lg:w-auto">
                  Start Analysis
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>
            <RadiologyImageCard
              src={radiologyImages.chestXrayReview}
              alt="Radiologist reviewing chest X-ray for graph mapping"
              label="Graph evidence"
              caption="Claims connected to image findings"
              className="min-h-[280px]"
            />
          </div>

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

          <KnowledgeGraph />
        </div>
      </section>
    </Layout>
  );
};

export default KnowledgeGraphPage;
