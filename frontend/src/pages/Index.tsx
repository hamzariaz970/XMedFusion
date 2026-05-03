import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";
import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabaseClient";
import {
  Activity,
  ArrowRight,
  Brain,
  CheckCircle2,
  FileSearch,
  FileText,
  Network,
  ShieldCheck,
  Upload,
  Users,
} from "lucide-react";

const stats = [
  { value: "4+", label: "AI Agents" },
  { value: "95%", label: "Evidence Focus" },
  { value: "24/7", label: "AI Assistance" },
  { value: "100+", label: "Clinical Signals" },
];

const services = [
  {
    title: "Report Synthesis",
    copy: "Turn X-ray and CT inputs into structured findings, impression, and recommendation sections.",
    icon: FileText,
    image: radiologyImages.laptopReview,
  },
  {
    title: "Virtual Review",
    copy: "Review AI output with explainability, heatmaps, and human-in-the-loop feedback.",
    icon: FileSearch,
    image: radiologyImages.teamReview,
    featured: true,
  },
  {
    title: "Knowledge Discovery",
    copy: "Connect anatomy, observations, and diagnostic claims through a clinical graph.",
    icon: Network,
    image: radiologyImages.chestXrayReview,
  },
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
          <div className="figma-hero relative min-h-[calc(100vh-5rem)] overflow-hidden rounded-none p-7 text-white md:min-h-[620px] md:rounded-[38px] md:p-16">
            <div className="relative z-10 max-w-2xl animate-slide-up">
              <h1 className="max-w-xl text-4xl font-extrabold leading-[1.05] tracking-tight md:text-6xl">
                Your Digital Gateway To{" "}
                <span className="text-primary">Evidence-Linked</span> Radiology AI
              </h1>
              <p className="mt-7 max-w-lg text-lg leading-8 text-white/90">
                Generate transparent reports, inspect AI reasoning, and guide each diagnosis with clinician-ready evidence.
              </p>
              <div className="mt-7 flex flex-col gap-3 sm:flex-row">
                <Link to={session ? "/upload" : "/login"}>
                  <Button variant="glass" size="lg" className="w-full rounded-full bg-white text-foreground hover:bg-white/90 sm:w-auto px-8">
                    Upload Scan
                  </Button>
                </Link>
                <Link to={session ? "/dashboard" : "/login"}>
                  <Button size="lg" className="w-full rounded-full bg-primary text-white hover:bg-primary/90 sm:w-auto px-8">
                    Dashboard
                  </Button>
                </Link>
              </div>

              <RadiologyImageCard
                src={radiologyImages.laptopReview}
                alt="Radiologists reviewing an X-ray on a laptop"
                label="Live report review"
                caption="AI draft + clinician approval"
                className="mt-8 h-60 md:hidden"
              />

              <div className="mt-10 w-full max-w-[340px] rounded-[24px] bg-white/10 border border-white/20 p-5 text-white shadow-glow backdrop-blur-md">
                <div className="flex items-center gap-4">
                  <div className="h-12 w-12 rounded-2xl bg-white/20 flex items-center justify-center">
                    <ShieldCheck className="h-6 w-6" />
                  </div>
                  <div>
                    <p className="text-sm font-bold">Clinical Evidence Layer</p>
                    <p className="text-xs text-white/70">Verified by Multi-Agent Workflow</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="absolute right-8 top-10 hidden h-full w-[48%] md:block">
              <RadiologyImageCard
                src={radiologyImages.laptopReview}
                alt="Radiologists reviewing an X-ray on a laptop"
                label="Live report review"
                caption="AI draft + clinician approval"
                className="absolute right-0 top-8 h-[410px] w-[92%] animate-drift"
              />
              <div className="report-glass-panel absolute left-0 top-24 w-64 animate-drift-delayed">
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-xs font-bold uppercase text-muted-foreground">Report status</span>
                  <span className="report-pulse-dot" />
                </div>
                <p className="text-lg font-extrabold text-foreground">Findings generated</p>
                <div className="mt-4 space-y-2">
                  <span className="block h-2 rounded-full bg-primary/30" />
                  <span className="block h-2 w-4/5 rounded-full bg-primary/20" />
                  <span className="block h-2 w-2/3 rounded-full bg-primary/20" />
                </div>
              </div>
              <div className="report-glass-panel absolute bottom-24 right-10 w-72">
                <div className="flex items-center gap-3">
                  <Brain className="h-9 w-9 text-primary" />
                  <div>
                    <p className="font-extrabold text-foreground">Evidence linked</p>
                    <p className="text-xs text-muted-foreground">Scan regions mapped to report text</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="figma-container figma-section">
          <div className="grid gap-10 md:grid-cols-[1.1fr_0.9fr] md:items-start">
            <div>
              <p className="mb-14 text-2xl font-semibold">About Us.</p>
              <h2 className="max-w-2xl text-4xl font-extrabold leading-tight tracking-tight md:text-6xl">
                Transforming The Way You <span className="text-primary">Access</span> Medical Imaging Care
              </h2>
            </div>
            <p className="text-lg leading-8 text-muted-foreground md:pt-20">
              XMedFusion is an AI-assisted radiology workspace designed to make imaging analysis clearer, faster, and safer. It connects reports, scan evidence, knowledge graphs, and physician review in one trusted flow.
            </p>
          </div>

          <div className="mt-16 grid grid-cols-2 gap-8 md:grid-cols-4">
            {stats.map((stat, index) => (
              <div key={stat.label} className="animate-slide-up" style={{ animationDelay: `${index * 90}ms` }}>
                <p className="figma-stat">{stat.value}</p>
                <p className="mt-3 text-sm text-muted-foreground">{stat.label}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="figma-container figma-section">
          <p className="mb-12 text-2xl font-semibold">Why Choose Us.</p>
          <div className="grid gap-8">
            <div className="grid gap-8 md:grid-cols-[0.9fr_1.1fr] md:items-end">
              <p className="max-w-md text-base leading-7 text-muted-foreground md:order-1 md:self-end">
                Get immediate support with agent-guided report synthesis, evidence links, and clinical review tools.
              </p>
              <h2 className="text-right text-4xl font-extrabold leading-tight tracking-tight md:text-5xl">
                Smart <span className="text-primary">24/7<br />AI Assistance</span><br />For Instant Support
              </h2>
            </div>
            <div className="relative h-[280px] overflow-hidden rounded-[34px] shadow-card">
              <RadiologyImageCard
                src={radiologyImages.digitalConsult}
                alt="Doctors reviewing an X-ray digitally during consultation"
                label="Scan intake ready"
                caption="Vision + retrieval + KG"
                className="h-full"
              />
              <div className="report-glass-panel absolute right-8 top-8 hidden w-72 md:block">
                <div className="flex items-center gap-3">
                  <Brain className="h-8 w-8 text-primary" />
                  <div>
                    <p className="font-bold">Agent triage</p>
                    <p className="text-xs text-muted-foreground">Processing image evidence</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid gap-8 pt-8 md:grid-cols-[0.8fr_1.2fr] md:items-center">
              <div>
                <h2 className="text-4xl font-extrabold leading-tight tracking-tight md:text-5xl">
                  Experienced Doctors You Can <span className="text-primary">Trust</span>
                </h2>
                <p className="mt-28 max-w-sm text-base leading-7 text-muted-foreground">
                  Keep radiologists in control with editable reports, expert labels, and feedback loops for better models.
                </p>
              </div>
              <div className="relative min-h-[430px]">
                <RadiologyImageCard
                  src={radiologyImages.patientExplanation}
                  alt="Doctor explaining an X-ray to a patient"
                  label="Human approved"
                  caption="Editable reports before final sign-off"
                  className="blob-card absolute inset-0"
                />
                <div className="absolute left-8 top-8 hidden rounded-[24px] bg-white/90 p-4 shadow-card backdrop-blur md:block">
                  <div className="flex items-center gap-3">
                    <CheckCircle2 className="h-8 w-8 text-primary" />
                    <div>
                      <p className="font-bold">Clinician verified</p>
                      <p className="text-xs text-muted-foreground">Every report stays reviewable</p>
                    </div>
                  </div>
                </div>
                <div className="absolute bottom-14 right-10 circle-button">
                  <ArrowRight className="h-8 w-8" />
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="figma-container figma-section">
          <div className="mb-16 grid gap-8 md:grid-cols-2 md:items-end">
            <div>
              <p className="mb-14 text-2xl font-semibold">Our Doctors.</p>
              <h2 className="text-4xl font-extrabold leading-tight tracking-tight md:text-6xl">
                Our <span className="text-primary">Professional</span><br />Expert
              </h2>
            </div>
            <p className="max-w-md text-right text-lg leading-8 text-muted-foreground md:justify-self-end">
              We support qualified clinicians with radiology-focused automation, transparent evidence, and human approval tools.
            </p>
          </div>

          <div className="grid gap-8 overflow-hidden md:grid-cols-[1fr_1fr_0.7fr]">
            <div className="blue-panel min-h-[420px] p-10">
              <h3 className="text-4xl font-semibold">Radiology Lead</h3>
              <p className="mt-2 text-xl text-white/80">Clinical review team.</p>
              <p className="mt-12 max-w-sm text-lg leading-7 text-white/80">
                Expert physicians can approve, correct, and strengthen every AI-generated report before it becomes part of the patient record.
              </p>
              <div className="absolute bottom-10 left-10 flex gap-4 text-white/90">
                <span className="h-8 w-8 rounded-full border border-white/50 text-center leading-8">f</span>
                <span className="h-8 w-8 rounded-full border border-white/50 text-center leading-8">x</span>
                <span className="h-8 w-8 rounded-full border border-white/50 text-center leading-8">in</span>
              </div>
            </div>
            <DoctorVisual name="Evidence first" image={radiologyImages.teamReview} />
            <DoctorVisual name="Human approved" image={radiologyImages.neuroReview} compact />
          </div>
        </section>

        <section className="figma-section bg-white/65">
          <div className="figma-container grid gap-10 md:grid-cols-[0.85fr_1.15fr] md:items-center">
            <div>
              <p className="mb-14 text-2xl font-semibold">User Experience.</p>
              <h2 className="text-4xl font-extrabold leading-tight tracking-tight md:text-6xl">
                What Doctors<br />Are Saying
              </h2>
              <p className="mt-16 max-w-sm text-lg leading-8 text-muted-foreground">
                Radiologists across clinical centers rely on XMedFusion for faster reporting and verifiable evidence links.
              </p>
              <Link to={session ? "/upload" : "/login"} className="mt-10 inline-flex circle-button">
                <ArrowRight className="h-8 w-8" />
              </Link>
            </div>
            <div className="relative min-h-[390px]">
              <div className="blue-panel ml-auto min-h-[330px] max-w-[580px] p-12">
                <p className="text-7xl font-black leading-none text-white">“</p>
                <h3 className="mt-2 max-w-md text-4xl font-semibold leading-tight">Secure, fast, and so convenient.</h3>
                <p className="mt-5 max-w-md text-lg leading-7 text-white/80">
                  The diagnostic workspace keeps scan evidence, AI reasoning, and physician review in one place.
                </p>
                <p className="absolute bottom-14 right-14 text-2xl font-semibold">Clinical User</p>
              </div>
              <RadiologyImageCard
                src={radiologyImages.chestXrayReview}
                alt="Radiologist examining chest X-ray"
                className="absolute bottom-10 left-8 h-32 w-36 rounded-[28px]"
                scanLine={false}
              />
            </div>
          </div>
          <div className="mt-16 bg-muted/80 py-8">
            <div className="figma-container grid gap-6 md:grid-cols-3">
              {[
                ["25", "years of research practice."],
                ["Quality", "Evidence links at every step."],
                ["24/7", "AI support for clinical teams."],
              ].map(([big, copy]) => (
                <div key={copy} className="flex items-center gap-5">
                  <div className="flex h-20 w-20 items-center justify-center rounded-full bg-white shadow-sm">
                    <ShieldCheck className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold">{big}</p>
                    <p className="text-muted-foreground">{copy}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="figma-container figma-section">
          <div className="mb-14 grid gap-8 md:grid-cols-2 md:items-end">
            <div>
              <p className="mb-10 text-2xl font-semibold">Our Services</p>
              <h2 className="max-w-2xl text-4xl font-extrabold leading-tight tracking-tight md:text-6xl">
                We Will Serve You With Healthcare Services
              </h2>
            </div>
            <div className="space-y-8 md:text-right">
              <p className="text-lg leading-8 text-muted-foreground">
                A complete workspace for interpretation, review, and transparent clinical delivery.
              </p>
              <Link to={session ? "/upload" : "/login"}>
                <Button variant="outline" size="lg" className="rounded-full px-8">
                  Try Diagnosis
                  <ArrowRight className="h-5 w-5" />
                </Button>
              </Link>
            </div>
          </div>

          <div className="grid gap-6 md:grid-cols-3">
            {services.map((service) => (
              <div
                key={service.title}
                className={service.featured ? "rounded-[26px] bg-primary p-7 text-white shadow-card transition-all duration-300 hover:-translate-y-2" : "rounded-[26px] border border-border bg-white p-7 shadow-sm transition-all duration-300 hover:-translate-y-2 hover:shadow-card"}
              >
                <div className="mb-8 flex items-start justify-between gap-4">
                  <h3 className="max-w-[220px] text-4xl font-extrabold leading-tight">{service.title}</h3>
                  <span className={service.featured ? "flex h-10 w-10 items-center justify-center rounded-full bg-white text-primary" : "flex h-10 w-10 items-center justify-center rounded-full bg-foreground text-white"}>
                    <ArrowRight className="-rotate-45" />
                  </span>
                </div>
                <p className={service.featured ? "text-lg leading-7 text-white/80" : "text-lg leading-7 text-muted-foreground"}>{service.copy}</p>
                <ServiceVisual image={service.image} title={service.title} featured={service.featured} />
              </div>
            ))}
          </div>
        </section>

        <section className="figma-container pb-20">
          <div className="relative overflow-hidden rounded-[28px] bg-primary p-10 text-white md:p-16">
            <div className="relative z-10 max-w-2xl">
              <h2 className="text-4xl font-extrabold leading-tight md:text-6xl">
                Schedule Your Analysis With Us
              </h2>
              <p className="mt-6 max-w-2xl text-lg leading-8 text-white/80">
                Choose your patient, upload a scan, and let XMedFusion assemble a transparent report ready for clinical review.
              </p>
              <Link to={session ? "/upload" : "/login"} className="mt-9 inline-flex">
                <Button variant="glass" size="lg" className="rounded-full bg-white text-foreground hover:bg-white/90 px-10">
                  Upload Scan
                  <ArrowRight className="h-5 w-5" />
                </Button>
              </Link>
            </div>
            <RadiologyImageCard
              src={radiologyImages.teamReview}
              alt="Radiology team reviewing imaging"
              className="absolute right-8 top-8 hidden h-56 w-72 rotate-3 rounded-[30px] md:block"
              scanLine={false}
            />
          </div>
        </section>
      </div>
    </Layout>
  );
};

const DoctorVisual = ({ name, image, compact }: { name: string; image: string; compact?: boolean }) => (
  <div className={compact ? "soft-image-card group relative min-h-[420px] min-w-[260px]" : "soft-image-card group relative min-h-[420px]"}>
    <img src={image} alt={`${name} radiology review`} className="absolute inset-0 h-full w-full object-cover transition-transform duration-700 group-hover:scale-105" loading="lazy" />
    <div className="absolute inset-0 bg-gradient-to-t from-clinical-ink/70 via-transparent to-transparent" />
    <div className="absolute bottom-8 left-8 right-8 flex items-center gap-4 rounded-[22px] bg-white/90 p-4 shadow-sm backdrop-blur">
      <Users className="h-9 w-9 text-primary" />
      <div>
        <p className="font-bold">{name}</p>
        <p className="text-sm text-muted-foreground">Specialist review</p>
      </div>
    </div>
  </div>
);

const ServiceVisual = ({ image, title, featured }: { image: string; title: string; featured?: boolean }) => (
  <RadiologyImageCard
    src={image}
    alt={`${title} workflow`}
    className={featured ? "mt-10 h-48 rounded-[20px]" : "mt-10 h-48 rounded-[20px]"}
    overlayClassName={featured ? "bg-white/95" : undefined}
    scanLine={featured}
  />
);

export default Index;
