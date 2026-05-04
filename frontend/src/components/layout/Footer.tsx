import { Github, Linkedin, Mail, Plus } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export const Footer = () => {
  return (
    <footer className="mt-auto bg-white text-foreground">
      <div className="figma-container grid gap-10 py-16 md:grid-cols-[1.5fr_0.8fr_0.8fr_auto]">
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

        <div>
          <h3 className="mb-5 font-semibold">Connect</h3>
          <div className="space-y-5">
            <div>
              <p className="text-sm font-medium text-foreground flex items-center gap-2 mb-2">
                <Mail className="h-4 w-4 text-muted-foreground" /> Email Us
              </p>
              <ul className="space-y-2 text-sm text-muted-foreground ml-6">
                <li className="select-all hover:text-primary transition-colors cursor-text">hriaz.bscs22seecs@seecs.edu.pk</li>
                <li className="select-all hover:text-primary transition-colors cursor-text">aharoon.bscs22seecs@seecs.edu.pk</li>
                <li className="select-all hover:text-primary transition-colors cursor-text">mbaig.bscs22seecs@seecs.edu.pk</li>
              </ul>
            </div>
            
            <div>
              <p className="text-sm font-medium text-foreground flex items-center gap-2 mb-2">
                <Linkedin className="h-4 w-4 text-muted-foreground" /> LinkedIn
              </p>
              <ul className="space-y-2 text-sm text-muted-foreground ml-6">
                <li><a href="https://linkedin.com/in/hamzariaz970" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors underline-offset-4 hover:underline">hamzariaz970</a></li>
                <li><a href="https://linkedin.com/in/maha-baig-95649b148" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors underline-offset-4 hover:underline">maha-baig-95649b148</a></li>
                <li><a href="https://linkedin.com/in/arham-haroon-7a6bb8243" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors underline-offset-4 hover:underline">arham-haroon-7a6bb8243</a></li>
              </ul>
            </div>

            <div>
              <p className="text-sm font-medium text-foreground flex items-center gap-2 mb-2">
                <Github className="h-4 w-4 text-muted-foreground" /> GitHub
              </p>
              <ul className="text-sm text-muted-foreground ml-6">
                <li><a href="https://github.com/hamzariaz970/XMedFusion" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors underline-offset-4 hover:underline">hamzariaz970/XMedFusion</a></li>
              </ul>
            </div>
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
