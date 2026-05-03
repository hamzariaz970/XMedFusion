import { Github, Linkedin, Mail, Plus } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export const Footer = () => {
  return (
    <footer className="mt-auto bg-white text-foreground">
      <div className="figma-container grid gap-10 py-16 md:grid-cols-[1.2fr_0.8fr_0.8fr_0.8fr]">
        <div>
          <Link to="/" className="mb-8 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-600 to-primary">
              <Plus className="h-6 w-6 stroke-[4] text-white" />
            </div>
            <span className="text-2xl font-extrabold">XMedFusion</span>
          </Link>
          <h2 className="max-w-sm text-2xl font-bold leading-tight tracking-tight">
            Advancing radiology with <span className="text-primary">transparent AI.</span>
          </h2>
        </div>

        <div>
          <h3 className="mb-5 font-semibold">Workspace</h3>
          <ul className="space-y-3 text-sm text-muted-foreground">
            <li><Link to="/dashboard" className="transition-colors hover:text-primary">Dashboard</Link></li>
            <li><Link to="/patients" className="transition-colors hover:text-primary">Patient Registry</Link></li>
            <li><Link to="/upload" className="transition-colors hover:text-primary">Upload</Link></li>
          </ul>
        </div>

        <div>
          <h3 className="mb-5 font-semibold">Evidence</h3>
          <ul className="space-y-3 text-sm text-muted-foreground">
            <li><Link to="/knowledge-graph" className="transition-colors hover:text-primary">Knowledge Graph</Link></li>
            <li><Link to="/explainability" className="transition-colors hover:text-primary">Explainability</Link></li>
          </ul>
        </div>

        <div className="flex flex-col items-end justify-start">
          <h3 className="mb-5 font-semibold w-full text-right">Connect</h3>
          <div className="flex items-center gap-3">
            {[Github, Mail, Linkedin].map((Icon, index) => (
              <a key={index} href="#" className="flex h-10 w-10 items-center justify-center rounded-full border border-border text-foreground transition-colors hover:bg-primary hover:text-white hover:border-primary">
                <Icon className="h-4 w-4" />
              </a>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-primary py-7 text-white">
        <div className="figma-container flex flex-col items-center justify-between gap-5 md:flex-row">
          <p className="text-sm text-white/75">© 2026 XMedFusion AI Radiology. All Rights Reserved.</p>
          <div className="flex gap-3">
            <Button variant="outline" size="sm" className="rounded-full border-white/30 bg-transparent text-white hover:bg-white hover:text-primary">
              Terms of Service
            </Button>
            <Button variant="outline" size="sm" className="rounded-full border-white/30 bg-transparent text-white hover:bg-white hover:text-primary">
              Privacy Policy
            </Button>
          </div>
        </div>
      </div>
    </footer>
  );
};
