import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";
import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabaseClient";
import architectureImage from "@/assets/xmedfusion-architecture.png";
import {
  ArrowRight,
  Brain,
  CheckCircle2,
  FileText,
  Microscope,
  Network,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Stethoscope,
} from "lucide-react";

const headlineMetrics = [
  { label: "BLEU-1", baseline: "0.0493", value: "0.3359" },
  { label: "ROUGE-L", baseline: "0.0863", value: "0.2440" },
  { label: "Consistency", baseline: "2.38", value: "7.80" },
  { label: "Accuracy", baseline: "2.34", value: "6.93" },
];

const agentStages = [
  {
    title: "Vision Agent",
    icon: ScanSearch,
    copy:
      "Extracts dense image-grounded descriptions from lung fields, pleura, mediastinum, and cardiac silhouette instead of relying on a single black-box pass.",
  },
  {
    title: "KG Agent",
    icon: Network,
    copy:
      "Builds a RadGraph-compliant knowledge graph from anatomy and observation nodes so clinical facts stay explicit and checkable.",
  },
  {
    title: "Retrieval & Draft Agent",
    icon: FileText,
    copy:
      "Retrieves top-k similar cases to supply contextual scaffolding, reducing stylistic drift while keeping retrieval separate from final evidence control.",
  },
  {
    title: "Synthesis Agent",
    icon: Brain,
    copy:
      "Uses MedGemma 1.5:4B to iteratively reconcile vision evidence, knowledge graph facts, and retrieved context into a clinically coherent report.",
  },
];

const strengths = [
  "Structured perception replaces single-pass generation with explicit intermediate evidence.",
  "Knowledge graph control reduces unsupported diagnostic statements and cross-modal inconsistency.",
  "Evidence-prioritized synthesis resolves conflicts across agents before the final report is produced.",
  "Explainability overlays and traceable findings let radiologists verify what the system actually saw.",
];

const explainabilityPoints = [
  "Visual grounding overlays expose the scan regions behind generated findings.",
  "Every final statement is traceable to upstream agent outputs rather than hidden decoder behavior.",
  "The knowledge graph acts as a control signal that blocks hallucinated claims.",
  "Radiologists can review evidence, report text, and explanations in one workflow.",
];

const Index = () => {
  const [session, setSession] = useState<any>(null);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });

    return () => subscription.unsubscribe();
  }, []);

  return (
    <Layout>
      <div className="figma-page">
        <section className="figma-container px-0 pt-0 md:px-8 md:pt-6 lg:px-10 xl:px-14">
          <div className="figma-hero relative overflow-hidden rounded-none p-7 text-white md:rounded-[38px] md:p-16">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.2),transparent_26%)]" />
            <div className="absolute right-8 top-8 z-10 hidden md:block lg:right-10 lg:top-10">
              <div className="animate-drift w-[320px] lg:w-[920px]">
                <div className="overflow-hidden rounded-[30px] border border-white/20 bg-white/8 shadow-card backdrop-blur-sm transition-all duration-300 hover:scale-[1.04] hover:shadow-2xl hover:border-white/35 cursor-default">
                  <img
                    src={radiologyImages.laptopReview}
                    alt="Radiologists reviewing an X-ray on a laptop"
                    className="h-[320px] w-full object-cover object-center lg:h-[460px]"
                    loading="lazy"
                  />
                </div>
                <div className="mt-5 grid w-full grid-cols-1 gap-4 lg:grid-cols-3">
                  <div className="group flex flex-col items-center justify-center rounded-[24px] border border-white/30 bg-white/95 px-4 py-5 text-center shadow-[0_8px_30px_rgb(0,0,0,0.08)] backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:shadow-[0_20px_40px_rgb(0,0,0,0.12)] cursor-default">
                    <div className="mb-3 rounded-full bg-primary/10 p-2.5 text-primary transition-transform duration-300 group-hover:scale-110 group-hover:bg-primary/15">
                      <ShieldCheck className="h-6 w-6" />
                    </div>
                    <p className="text-sm font-extrabold text-primary md:text-base">Transparent</p>
                    <p className="mt-1 text-xs font-semibold leading-relaxed text-muted-foreground">Agent outputs stay reviewable</p>
                  </div>
                  <div className="group flex flex-col items-center justify-center rounded-[24px] border border-white/30 bg-white/95 px-4 py-5 text-center shadow-[0_8px_30px_rgb(0,0,0,0.08)] backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:shadow-[0_20px_40px_rgb(0,0,0,0.12)] cursor-default">
                    <div className="mb-3 rounded-full bg-primary/10 p-2.5 text-primary transition-transform duration-300 group-hover:scale-110 group-hover:bg-primary/15">
                      <Network className="h-6 w-6" />
                    </div>
                    <p className="text-sm font-extrabold text-primary md:text-base">Multi-Agent</p>
                    <p className="mt-1 text-xs font-semibold leading-relaxed text-muted-foreground">Decomposed clinical reasoning</p>
                  </div>
                  <div className="group flex flex-col items-center justify-center rounded-[24px] border border-white/30 bg-white/95 px-4 py-5 text-center shadow-[0_8px_30px_rgb(0,0,0,0.08)] backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:shadow-[0_20px_40px_rgb(0,0,0,0.12)] cursor-default">
                    <div className="mb-3 rounded-full bg-primary/10 p-2.5 text-primary transition-transform duration-300 group-hover:scale-110 group-hover:bg-primary/15">
                      <CheckCircle2 className="h-6 w-6" />
                    </div>
                    <p className="text-sm font-extrabold text-primary md:text-base">Grounded</p>
                    <p className="mt-1 text-xs font-semibold leading-relaxed text-muted-foreground">No black-box generation</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="relative z-10">
              <div className="animate-slide-up md:max-w-[55%]">
                <div className="eyebrow animate-fade-in border-white/20 bg-white/10 text-white shadow-none">
                  NUST Final Year Project
                </div>
                <h1 className="mt-6 max-w-4xl animate-slide-up text-4xl font-extrabold leading-[1.02] tracking-tight md:text-6xl">
                  XMedFusion brings transparent, evidence-grounded AI to radiology report generation.
                </h1>
                <p className="mt-6 max-w-2xl animate-slide-up text-lg leading-8 text-white/88 stagger-1">
                  Our multi-agent framework decomposes chest X-ray reporting into visual perception, knowledge graph construction, retrieval-guided drafting, and evidence-prioritized synthesis to reduce hallucinations and improve diagnostic reliability.
                </p>
                <div className="mt-8 flex flex-wrap gap-3 animate-slide-up stagger-2">
                  <span className="medical-chip border-white/15 bg-white/10 text-white">IU X-ray benchmark</span>
                  <span className="medical-chip border-white/15 bg-white/10 text-white">BioMedCLIP + MedGemma 1.5:4B</span>
                  <span className="medical-chip border-white/15 bg-white/10 text-white">Explainable report generation</span>
                </div>
                <div className="mt-8 flex flex-col gap-3 animate-slide-up stagger-3 sm:flex-row">
                  <Link to={session ? "/upload" : "/login"}>
                    <Button variant="glass" size="lg" className="w-full rounded-full bg-white px-8 text-foreground hover:bg-white/90 sm:w-auto">
                      Upload Scan
                    </Button>
                  </Link>
                  <Link to={session ? "/dashboard" : "/login"}>
                    <Button size="lg" className="w-full rounded-full bg-primary px-8 text-white hover:bg-primary/90 sm:w-auto">
                      Open Workspace
                    </Button>
                  </Link>
                </div>
                <RadiologyImageCard
                  src={radiologyImages.laptopReview}
                  alt="Radiologists reviewing an X-ray on a laptop"
                  label="Live report review"
                  caption="Evidence-linked findings for clinician approval"
                  className="mt-8 h-60 animate-slide-up md:hidden"
                />
                <div className="mt-5 grid w-full grid-cols-1 gap-4 animate-slide-up md:hidden">
                  <div className="group flex flex-col items-center justify-center rounded-[24px] border border-white/30 bg-white/95 px-4 py-5 text-center shadow-[0_8px_30px_rgb(0,0,0,0.08)] backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:shadow-[0_20px_40px_rgb(0,0,0,0.12)] cursor-default">
                    <div className="mb-3 rounded-full bg-primary/10 p-2.5 text-primary transition-transform duration-300 group-hover:scale-110 group-hover:bg-primary/15">
                      <ShieldCheck className="h-6 w-6" />
                    </div>
                    <p className="text-sm font-extrabold text-primary md:text-base">Transparent</p>
                    <p className="mt-1 text-xs font-semibold leading-relaxed text-muted-foreground">Agent outputs stay reviewable</p>
                  </div>
                  <div className="group flex flex-col items-center justify-center rounded-[24px] border border-white/30 bg-white/95 px-4 py-5 text-center shadow-[0_8px_30px_rgb(0,0,0,0.08)] backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:shadow-[0_20px_40px_rgb(0,0,0,0.12)] cursor-default">
                    <div className="mb-3 rounded-full bg-primary/10 p-2.5 text-primary transition-transform duration-300 group-hover:scale-110 group-hover:bg-primary/15">
                      <Network className="h-6 w-6" />
                    </div>
                    <p className="text-sm font-extrabold text-primary md:text-base">Multi-Agent</p>
                    <p className="mt-1 text-xs font-semibold leading-relaxed text-muted-foreground">Decomposed clinical reasoning</p>
                  </div>
                  <div className="group flex flex-col items-center justify-center rounded-[24px] border border-white/30 bg-white/95 px-4 py-5 text-center shadow-[0_8px_30px_rgb(0,0,0,0.08)] backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:shadow-[0_20px_40px_rgb(0,0,0,0.12)] cursor-default">
                    <div className="mb-3 rounded-full bg-primary/10 p-2.5 text-primary transition-transform duration-300 group-hover:scale-110 group-hover:bg-primary/15">
                      <CheckCircle2 className="h-6 w-6" />
                    </div>
                    <p className="text-sm font-extrabold text-primary md:text-base">Grounded</p>
                    <p className="mt-1 text-xs font-semibold leading-relaxed text-muted-foreground">No black-box generation</p>
                  </div>
                </div>
              </div>

              <div className="mt-8 grid gap-4 md:mt-[380px] lg:mt-16 xl:mt-24 2xl:mt-32">
                <div className="report-glass-panel animate-float-soft border-white/15 bg-white/10 text-white transition-all duration-300 hover:scale-[1.02] hover:bg-white/15 hover:shadow-xl cursor-default">
                  <div className="flex items-start gap-4">
                    <ShieldCheck className="mt-1 h-8 w-8 text-white transition-transform duration-300 group-hover:rotate-6" />
                    <div>
                      <p className="text-xl font-bold">Why it outperforms traditional models</p>
                      <p className="mt-2 text-base leading-7 text-white/80">
                        Traditional end-to-end VLMs often miss subtle regional cues and produce unsupported findings. XMedFusion inserts explicit evidence checks between perception and generation.
                      </p>
                    </div>
                  </div>
                </div>
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {headlineMetrics.map((metric, index) => (
                    <div key={metric.label} className="animate-slide-up h-full" style={{ animationDelay: `${index * 90}ms` }}>
                      <div className="h-full rounded-[24px] border border-white/15 bg-white/10 p-5 shadow-glow backdrop-blur-md transition-all duration-300 hover:scale-105 hover:bg-white/18 hover:shadow-xl cursor-default">
                        <p className="text-sm font-bold uppercase tracking-[0.2em] text-white/60">{metric.label}</p>
                        <div className="mt-3 flex items-end justify-between gap-3">
                          <div>
                            <p className="text-3xl font-extrabold">{metric.value}</p>
                            <p className="text-sm text-white/70">XMedFusion</p>
                          </div>
                          <div className="text-right">
                            <p className="text-base font-semibold text-white/75">{metric.baseline}</p>
                            <p className="text-sm text-white/60">LLaVA-Med 1.5</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="figma-container figma-section">
          <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
            <div className="surface-card animate-slide-up p-8 md:p-10">
              <div className="eyebrow">Project Focus</div>
              <h2 className="mt-6 animate-sheen pb-4 text-3xl font-extrabold leading-tight md:text-5xl">
                Built for faster, more reliable, and more interpretable radiology reporting.
              </h2>
              <p className="mt-8 max-w-xl text-base leading-8 text-muted-foreground">
                The project targets automated generation of Findings and Impression sections for chest radiographs while preserving traceable evidence at each stage of the pipeline.
              </p>
              <RadiologyImageCard
                src={radiologyImages.digitalConsult}
                alt="Doctors digitally consulting over chest X-ray data"
                label="Clinical context"
                caption="Imaging backlogs and expert review needs"
                className="mt-8 h-64 animate-float-soft"
              />
            </div>

            <div className="grid gap-5 md:grid-cols-2">
              {[
                {
                  title: "Clinical problem",
                  icon: Stethoscope,
                  copy:
                    "Radiology reporting is time-intensive, imaging volumes are rising, and fatigue makes manual interpretation vulnerable to error.",
                },
                {
                  title: "Research gap",
                  icon: Microscope,
                  copy:
                    "Prior systems do not combine image-grounded evidence extraction, structured KG control, and iterative evidence-prioritized synthesis in one unified framework.",
                },
                {
                  title: "Core objective",
                  icon: Brain,
                  copy:
                    "Generate clinically useful reports with stronger visual grounding, reduced hallucinations, and explicit intermediate verification.",
                },
                {
                  title: "Evaluation setup",
                  icon: CheckCircle2,
                  copy:
                    "Benchmarked on the IU X-ray dataset with 2,068 training pairs and 590 test pairs using lexical and LLM-as-a-judge semantic metrics.",
                },
              ].map((item, index) => (
                <div key={item.title} className="animate-slide-up h-full" style={{ animationDelay: `${index * 80}ms` }}>
                  <div className="surface-card h-full p-6 transition-all duration-300 hover:scale-[1.04] hover:shadow-xl cursor-default">
                    <item.icon className="h-8 w-8 text-primary transition-transform duration-300 hover:scale-110" />
                    <h3 className="mt-5 text-2xl font-bold">{item.title}</h3>
                    <p className="mt-3 text-base leading-7 text-muted-foreground">{item.copy}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="figma-container figma-section">
          <div className="mb-10 flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
            <div>
              <div className="eyebrow">Agentic Pipeline</div>
              <h2 className="mt-5 animate-sheen text-4xl font-extrabold leading-tight tracking-tight md:text-5xl">
                Four specialized agents, one evidence-backed report.
              </h2>
            </div>
            <p className="max-w-2xl animate-slide-up text-base leading-8 text-muted-foreground">
              XMedFusion does not ask one model to do everything. Each component owns a distinct evidential role, which makes the final report easier to validate and more robust than traditional single-pass generation.
            </p>
          </div>

          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
            {agentStages.map((stage, index) => (
              <div key={stage.title} className="animate-slide-up h-full" style={{ animationDelay: `${index * 85}ms` }}>
                <div className="surface-card h-full p-6 transition-all duration-300 hover:scale-[1.04] hover:shadow-xl cursor-default">
                  <stage.icon className="h-8 w-8 text-primary transition-transform duration-300 hover:scale-110" />
                  <h3 className="mt-5 text-2xl font-bold">{stage.title}</h3>
                  <p className="mt-3 text-base leading-7 text-muted-foreground">{stage.copy}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 overflow-hidden rounded-[32px] border border-border/70 bg-white p-4 shadow-card animate-slide-up transition-all duration-300 hover:shadow-xl md:p-6">
            <div className="mb-5 flex items-center justify-between gap-4">
              <div>
                <p className="text-xl font-bold">Complete system architecture</p>
                <p className="text-base text-muted-foreground">
                  Directly adapted from the project presentation and embedded into the landing page.
                </p>
              </div>
              <span className="hidden rounded-full bg-primary/10 px-4 py-2 text-sm font-bold uppercase tracking-[0.18em] text-primary md:inline-flex">
                Stage 1 to Stage 3
              </span>
            </div>
            <div className="rounded-[26px] bg-[linear-gradient(180deg,rgba(237,244,255,0.9),rgba(255,255,255,0.95))] p-3">
              <img
                src={architectureImage}
                alt="XMedFusion multi-agent architecture showing preprocessing, parallel agentic processing, and synthesis with explainability"
                className="w-full rounded-[20px] border border-primary/10 bg-white object-cover"
                loading="lazy"
              />
            </div>
          </div>
        </section>

        <section className="figma-section bg-white/65">
          <div className="figma-container grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
            <div className="surface-card animate-slide-up p-8 md:p-10">
              <div className="eyebrow">Performance</div>
              <h2 className="mt-6 animate-sheen text-4xl font-extrabold leading-tight tracking-tight md:text-5xl">
                Stronger results than the baseline across lexical and semantic quality.
              </h2>
              <p className="mt-6 text-base leading-8 text-muted-foreground">
                On the IU X-ray test set, XMedFusion outperformed LLaVA-Med 1.5 on BLEU, ROUGE, METEOR, coverage, consistency, accuracy, style, and conciseness. The gains come from explicit evidence decomposition instead of direct end-to-end report generation.
              </p>
              <RadiologyImageCard
                src={radiologyImages.teamReview}
                alt="Clinical team reviewing radiology output"
                label="Benchmark review"
                caption="Human-readable comparisons with grounded evidence"
                className="mt-8 h-60 animate-float-soft"
              />
              <div className="mt-8 grid gap-4 sm:grid-cols-2">
                {[
                  "ROUGE-2 improved from 0.0213 to 0.1328.",
                  "METEOR improved from 0.0829 to 0.1708.",
                  "Coverage improved from 2.54 to 5.73.",
                  "Conciseness improved from 2.16 to 7.33.",
                ].map((line, index) => (
                  <div key={line} className="animate-slide-up" style={{ animationDelay: `${index * 70}ms` }}>
                    <div className="rounded-[22px] bg-primary/6 p-4 text-base font-medium leading-7 text-foreground transition-all duration-300 hover:scale-[1.03] hover:bg-primary/10 hover:shadow-md cursor-default">
                      {line}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="clinical-dark-panel animate-slide-up p-8 md:p-10">
              <div className="eyebrow border-white/15 bg-white/10 text-white shadow-none">Why It Works Better</div>
              <div className="mt-6 space-y-4">
                {strengths.map((point, index) => (
                  <div key={point} className="animate-slide-up" style={{ animationDelay: `${index * 75}ms` }}>
                    <div className="rounded-[22px] border border-white/10 bg-white/8 p-5 transition-all duration-300 hover:scale-[1.03] hover:bg-white/14 hover:border-white/20 hover:shadow-lg cursor-default">
                      <p className="text-base leading-7 text-white/86">{point}</p>
                    </div>
                  </div>
                ))}
                <RadiologyImageCard
                  src={radiologyImages.chestXrayReview}
                  alt="Radiologist examining chest X-ray"
                  label="Traditional vs XMedFusion"
                  caption="Grounded multi-stage reasoning improves reliability"
                  className="mt-6 h-56"
                  scanLine={false}
                />
              </div>
            </div>
          </div>
        </section>

        <section className="figma-container figma-section">
          <div className="grid gap-8 lg:grid-cols-[0.95fr_1.05fr]">
            <div className="clinical-dark-panel animate-slide-up p-8 md:p-10">
              <div className="eyebrow border-white/15 bg-white/10 text-white shadow-none">Explainability</div>
              <h2 className="mt-6 animate-sheen text-4xl font-extrabold leading-tight tracking-tight md:text-5xl">
                Transparent by design, not as an afterthought.
              </h2>
              <p className="mt-6 text-base leading-8 text-white/80">
                The system presentation emphasizes traceable evidence paths, visual grounding, and radiologist verification. That makes the homepage narrative align with the product itself: the report, the graph, and the explanation module all support one another.
              </p>
              <RadiologyImageCard
                src={radiologyImages.patientExplanation}
                alt="Doctor explaining radiology findings to a patient"
                label="Explainability in practice"
                caption="Evidence stays understandable beyond the model output"
                className="mt-8 h-72"
                scanLine={false}
              />
            </div>

            <div className="grid gap-5 md:grid-cols-2">
              {explainabilityPoints.map((point, index) => (
                <div key={point} className="animate-slide-up h-full" style={{ animationDelay: `${index * 80}ms` }}>
                  <div className="surface-card h-full p-6 transition-all duration-300 hover:scale-[1.04] hover:shadow-xl cursor-default">
                    <Sparkles className="h-7 w-7 text-primary transition-transform duration-300 hover:scale-110 hover:rotate-12" />
                    <p className="mt-4 text-base leading-7 text-muted-foreground">{point}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="figma-container pb-20">
          <div className="relative overflow-hidden rounded-[28px] bg-primary p-10 text-white md:p-16">
            <div className="relative z-10 max-w-3xl">
              <div className="eyebrow border-white/15 bg-white/10 text-white shadow-none">Project Summary</div>
              <h2 className="mt-6 animate-sheen text-4xl font-extrabold leading-tight md:text-6xl">
                XMedFusion moves beyond generic VLM reporting toward structured, trustworthy diagnostic AI.
              </h2>
              <p className="mt-6 max-w-2xl animate-slide-up text-lg leading-8 text-white/82">
                The home page now reflects the actual final year project: modular perception, neuro-symbolic reasoning, evidence-prioritized synthesis, and benchmarked gains in reliability and interpretability.
              </p>
              <Link to={session ? "/upload" : "/login"} className="mt-9 inline-flex">
                <Button variant="glass" size="lg" className="rounded-full bg-white px-10 text-foreground hover:bg-white/90">
                  Try the workflow
                  <ArrowRight className="h-5 w-5" />
                </Button>
              </Link>
            </div>
            <RadiologyImageCard
              src={radiologyImages.neuroReview}
              alt="Radiologist reviewing medical imaging"
              label="Clinical trust"
              caption="Built for reviewable AI-assisted reporting"
              className="absolute right-8 top-8 hidden h-60 w-72 rotate-3 md:block"
              scanLine={false}
            />
          </div>
        </section>
      </div>
    </Layout>
  );
};

export default Index;
