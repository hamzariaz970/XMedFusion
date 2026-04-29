import { Github, Linkedin, Mail, Plus } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export const Footer = () => {
  return (
    <footer className="mt-auto bg-white text-foreground">
      <div className="figma-container grid gap-10 py-16 md:grid-cols-[1.2fr_0.7fr_0.8fr_1.2fr]">
        <div>
          <Link to="/" className="mb-8 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-600 to-primary">
              <Plus className="h-6 w-6 stroke-[4] text-white" />
            </div>
            <span className="text-2xl font-extrabold">XMedFusion</span>
          </Link>
          <h2 className="max-w-sm text-4xl font-extrabold leading-tight tracking-tight">
            Join our team, Transform <span className="text-primary">healthcare.</span>
          </h2>
        </div>

        <div>
          <h3 className="mb-5 font-semibold">Menu</h3>
          <ul className="space-y-3 text-sm text-muted-foreground">
            <li><Link to="/" className="transition-colors hover:text-primary">About</Link></li>
            <li><Link to="/" className="transition-colors hover:text-primary">Blogs</Link></li>
            <li><Link to="/knowledge-graph" className="transition-colors hover:text-primary">How It Works</Link></li>
            <li><Link to="/image-mapping" className="transition-colors hover:text-primary">Testimonials</Link></li>
          </ul>
        </div>

        <div>
          <h3 className="mb-5 font-semibold">Product</h3>
          <ul className="space-y-3 text-sm text-muted-foreground">
            <li><Link to="/patients" className="transition-colors hover:text-primary">Patient Resources</Link></li>
            <li><Link to="/upload" className="transition-colors hover:text-primary">Appointments</Link></li>
            <li><Link to="/explainability" className="transition-colors hover:text-primary">Services</Link></li>
          </ul>
        </div>

        <div>
          <h3 className="mb-5 font-semibold">Subscribe our newsletter for update</h3>
          <div className="flex rounded-full border border-border p-1">
            <Input className="h-10 rounded-full border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0" placeholder="Email" />
            <Button className="h-10 rounded-full px-7">SEND</Button>
          </div>
        </div>
      </div>

      <div className="bg-primary py-7 text-white">
        <div className="figma-container flex flex-col items-center justify-between gap-5 md:flex-row">
          <div className="flex items-center gap-3">
            {[Github, Mail, Linkedin].map((Icon, index) => (
              <a key={index} href="#" className="flex h-9 w-9 items-center justify-center rounded-full border border-white/60 text-white/90 transition-colors hover:bg-white hover:text-primary">
                <Icon className="h-4 w-4" />
              </a>
            ))}
          </div>
          <p className="text-sm text-white/75">2026 All Rights Reserved</p>
          <div className="flex gap-3">
            <Button variant="outline" size="sm" className="rounded-full border-white/70 bg-transparent text-white hover:bg-white hover:text-primary">
              Terms & Condition
            </Button>
            <Button variant="outline" size="sm" className="rounded-full border-white/70 bg-transparent text-white hover:bg-white hover:text-primary">
              Privacy Policy
            </Button>
          </div>
        </div>
      </div>
    </footer>
  );
};
