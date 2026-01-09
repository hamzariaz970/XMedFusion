import { Activity, Github, Linkedin, Mail } from "lucide-react";
import { Link } from "react-router-dom";

export const Footer = () => {
  return (
    <footer className="bg-foreground text-background py-12 mt-auto">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-1 md:col-span-2">
            <Link to="/" className="flex items-center gap-2 mb-4">
              <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
                <Activity className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold">
                XMed<span className="text-primary">Fusion</span>
              </span>
            </Link>
            <p className="text-background/70 max-w-md">
              An agentic AI framework for generating clinically grounded radiology reports from medical imaging with full transparency and evidence linking.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2 text-background/70">
              <li><Link to="/" className="hover:text-primary transition-colors">Home</Link></li>
              <li><Link to="/upload" className="hover:text-primary transition-colors">Upload X-ray</Link></li>
              <li><Link to="/knowledge-graph" className="hover:text-primary transition-colors">Knowledge Graph</Link></li>
              {/* <li><Link to="/image-mapping" className="hover:text-primary transition-colors">Image Mapping</Link></li> */}
            </ul>
          </div>

          {/* Technologies */}
          <div>
            <h4 className="font-semibold mb-4">Technologies</h4>
            <ul className="space-y-2 text-background/70 text-sm">
              <li>PyTorch & Transformers</li>
              <li>ResNet-101 & Swin Transformer</li>
              <li>CLIP & Knowledge Graphs</li>
              <li>Multi-Agent Systems</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-background/20 mt-8 pt-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-background/60 text-sm">
            © 2025 XMedFusion. Advancing transparent AI for healthcare.
          </p>
          <div className="flex items-center gap-4">
            <a href="#" className="text-background/60 hover:text-primary transition-colors">
              <Github className="w-5 h-5" />
            </a>
            <a href="#" className="text-background/60 hover:text-primary transition-colors">
              <Linkedin className="w-5 h-5" />
            </a>
            <a href="#" className="text-background/60 hover:text-primary transition-colors">
              <Mail className="w-5 h-5" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};
