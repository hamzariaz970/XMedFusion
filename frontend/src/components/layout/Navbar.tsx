import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  Activity,
  Upload,
  Network,
  FileImage,
  Menu,
  X,
  Users,
  FileSearch,
} from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

const navItems = [
  { path: "/", label: "Home", icon: Activity },
  { path: "/upload", label: "Upload X-ray", icon: Upload },
  { path: "/explainability", label: "Explainability", icon: FileSearch },
  { path: "/knowledge-graph", label: "Knowledge Graph", icon: Network },
  { path: "/patients", label: "Patients", icon: Users },

  //{ path: "/image-mapping", label: "Image Mapping", icon: FileImage },
];

export const Navbar = () => {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-card/80 backdrop-blur-xl border-b border-border/50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group">
            <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-glow transition-transform group-hover:scale-105">
              <Activity className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="text-xl font-bold text-foreground">
              XMed<span className="text-primary">Fusion</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <Link key={item.path} to={item.path}>
                <Button
                  variant={
                    location.pathname === item.path ? "default" : "ghost"
                  }
                  size="sm"
                  className={cn(
                    "gap-2",
                    location.pathname === item.path && "shadow-glow"
                  )}
                >
                  <item.icon className="w-4 h-4" />
                  {item.label}
                </Button>
              </Link>
            ))}
          </div>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </Button>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden py-4 animate-slide-up">
            <div className="flex flex-col gap-2">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setIsOpen(false)}
                >
                  <Button
                    variant={
                      location.pathname === item.path ? "default" : "ghost"
                    }
                    className="w-full justify-start gap-2"
                  >
                    <item.icon className="w-4 h-4" />
                    {item.label}
                  </Button>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};
