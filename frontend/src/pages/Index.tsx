import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Link } from "react-router-dom";
import { 
  Activity, 
  Upload, 
  Brain, 
  FileText, 
  Network, 
  Shield, 
  Zap, 
  Eye,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  Layers
} from "lucide-react";

// UPDATED STATS TO REFLECT MODEL METRICS
const stats = [
  { value: "0.415", label: "BLEU-1 Score" },
  { value: "0.088", label: "BLEU-4 Score" },
  { value: "0.301", label: "METEOR Score" },
  { value: "0.303", label: "ROUGE-L F1" },
];

const features = [
  {
    icon: Eye,
    title: "Vision Agent",
    description: "Advanced abnormality detection using ResNet-101 and Swin Transformer for precise medical image analysis.",
  },
  {
    icon: Database,
    title: "Retrieval Agent",
    description: "Case-based reasoning system that retrieves similar cases for evidence-supported diagnoses.",
  },
  {
    icon: Brain,
    title: "LLM Agents",
    description: "Draft, Refiner, and Synthesis agents powered by T5 and FLAN-T5 for coherent report generation.",
  },
  {
    icon: Network,
    title: "Knowledge Graph",
    description: "Clinical knowledge graph storing structured findings, spatial metadata, and relationships.",
  },
  {
    icon: Shield,
    title: "Evidence Gate",
    description: "Verification mechanism that validates diagnostic claims against image-derived facts.",
  },
  {
    icon: Zap,
    title: "Real-time Processing",
    description: "Optimized pipeline for fast preprocessing and report generation with DICOM support.",
  },
];

const workflowSteps = [
  {
    step: "01",
    title: "Upload Image",
    description: "Upload chest X-ray or CT images in DICOM, JPEG, or PNG format.",
    icon: Upload,
  },
  {
    step: "02",
    title: "AI Analysis",
    description: "Multi-agent system processes the image for abnormality detection.",
    icon: Cpu,
  },
  {
    step: "03",
    title: "Knowledge Linking",
    description: "Findings are connected to the clinical knowledge graph.",
    icon: Network,
  },
  {
    step: "04",
    title: "Report Generation",
    description: "Evidence-linked radiology report with traceable findings.",
    icon: FileText,
  },
];

const Index = () => {
  return (
    <Layout>
      {/* Hero Section */}
      <section className="relative overflow-hidden hero-gradient py-20 lg:py-32">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-primary/10 rounded-full blur-3xl animate-pulse-slow" />
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-accent/10 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '2s' }} />
        </div>
        
        <div className="container mx-auto px-4 relative">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6 animate-fade-in">
              <Activity className="w-4 h-4" />
              AI-Powered Medical Imaging Analysis
            </div>
            
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-foreground mb-6 animate-slide-up">
              Transparent AI for{" "}
              <span className="gradient-text">Radiology Reports</span>
            </h1>
            
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-8 animate-slide-up" style={{ animationDelay: '0.1s' }}>
              An agentic AI framework generating clinically grounded reports from chest X-rays and CT images with full evidence linking and reduced hallucination.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-slide-up" style={{ animationDelay: '0.2s' }}>
              <Link to="/upload">
                <Button variant="hero" size="xl">
                  <Upload className="w-5 h-5" />
                  Upload X-ray
                </Button>
              </Link>
              <Link to="/knowledge-graph">
                <Button variant="outline" size="xl">
                  Explore Knowledge Graph
                  <ArrowRight className="w-5 h-5" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-card border-y border-border">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center animate-fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
                <div className="text-3xl md:text-4xl font-bold text-primary mb-2">{stat.value}</div>
                <div className="text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="py-20 lg:py-28">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-sm font-semibold text-primary uppercase tracking-wider mb-3">About Us</h2>
              <h3 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
                Transforming Medical Imaging with{" "}
                <span className="text-primary">AI Transparency</span>
              </h3>
              <p className="text-muted-foreground mb-6">
                XMedAgent combines state-of-the-art deep learning with clinical knowledge graphs to generate radiology reports that are not only accurate but fully traceable. Every diagnostic claim is verified against image-derived evidence.
              </p>
              
              <ul className="space-y-4">
                {[
                  "Multi-agent architecture for comprehensive analysis",
                  "Evidence gate mechanism reduces hallucination",
                  "Clinical knowledge graph for structured findings",
                  "Supports DICOM, JPEG, and PNG formats",
                ].map((item, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
                    <span className="text-foreground">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="relative">
              <div className="aspect-square rounded-3xl bg-gradient-to-br from-primary/20 to-accent/20 p-8 flex items-center justify-center">
                <div className="grid grid-cols-2 gap-4 w-full">
                  <div className="bg-card rounded-2xl p-6 shadow-card animate-float">
                    <Brain className="w-10 h-10 text-primary mb-3" />
                    <div className="font-semibold">AI Analysis</div>
                    <div className="text-sm text-muted-foreground">Deep Learning</div>
                  </div>
                  <div className="bg-card rounded-2xl p-6 shadow-card animate-float" style={{ animationDelay: '1s' }}>
                    <Network className="w-10 h-10 text-accent mb-3" />
                    <div className="font-semibold">Knowledge Graph</div>
                    <div className="text-sm text-muted-foreground">Structured Data</div>
                  </div>
                  <div className="bg-card rounded-2xl p-6 shadow-card animate-float" style={{ animationDelay: '2s' }}>
                    <Shield className="w-10 h-10 text-medical-success mb-3" />
                    <div className="font-semibold">Evidence Gate</div>
                    <div className="text-sm text-muted-foreground">Verification</div>
                  </div>
                  <div className="bg-card rounded-2xl p-6 shadow-card animate-float" style={{ animationDelay: '3s' }}>
                    <FileText className="w-10 h-10 text-medical-warning mb-3" />
                    <div className="font-semibold">Reports</div>
                    <div className="text-sm text-muted-foreground">Traceable</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Workflow Section */}
      <section className="py-20 lg:py-28 bg-muted/50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-sm font-semibold text-primary uppercase tracking-wider mb-3">How It Works</h2>
            <h3 className="text-3xl md:text-4xl font-bold text-foreground">
              Simple <span className="text-primary">Workflow</span>
            </h3>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {workflowSteps.map((step, index) => (
              <Card key={index} className="glass-card border-0 overflow-hidden group hover:shadow-card-hover transition-all duration-300">
                <CardContent className="p-6">
                  <div className="text-5xl font-bold text-primary/20 mb-4">{step.step}</div>
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                    <step.icon className="w-6 h-6 text-primary group-hover:text-primary-foreground" />
                  </div>
                  <h4 className="text-lg font-semibold text-foreground mb-2">{step.title}</h4>
                  <p className="text-muted-foreground text-sm">{step.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 lg:py-28">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-sm font-semibold text-primary uppercase tracking-wider mb-3">Features</h2>
            <h3 className="text-3xl md:text-4xl font-bold text-foreground">
              Powered by <span className="text-primary">Advanced AI</span>
            </h3>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <Card key={index} className="group hover:shadow-card-hover transition-all duration-300 border-border/50 hover:border-primary/30">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary transition-colors">
                    <feature.icon className="w-6 h-6 text-primary group-hover:text-primary-foreground" />
                  </div>
                  <h4 className="text-lg font-semibold text-foreground mb-2">{feature.title}</h4>
                  <p className="text-muted-foreground text-sm">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 lg:py-28 bg-primary text-primary-foreground">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Ready to Transform Your Radiology Workflow?
            </h2>
            <p className="text-primary-foreground/80 mb-8 text-lg">
              Upload your first X-ray and experience the power of evidence-linked AI report generation.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/upload">
                <Button variant="glass" size="xl" className="bg-primary-foreground text-primary hover:bg-primary-foreground/90">
                  <Upload className="w-5 h-5" />
                  Get Started Now
                </Button>
              </Link>
              <Link to="/knowledge-graph">
                <Button variant="outline" size="xl" className="border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10">
                  View Knowledge Graph
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
};

export default Index;