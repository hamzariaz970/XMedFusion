import { Link, useLocation, useNavigate } from "react-router-dom";
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
  LogIn,
  LogOut,
  ShieldCheck,
} from "lucide-react";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabaseClient";
import { usePatientContext } from "@/context/PatientContext";

const navItems = [
  { path: "/", label: "Home", icon: Activity, requiresAuth: false, requiresPatient: false },
  { path: "/patients", label: "Patients", icon: Users, requiresAuth: true, requiresPatient: false },
  { path: "/upload", label: "Upload", icon: Upload, requiresAuth: true, requiresPatient: true },
  { path: "/explainability", label: "Explainability", icon: FileSearch, requiresAuth: true, requiresPatient: true },

  //{ path: "/image-mapping", label: "Image Mapping", icon: FileImage, requiresAuth: true, requiresPatient: true },
];

export const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);
  const [session, setSession] = useState<any>(null);
  const { selectedPatient, setSelectedPatient } = usePatientContext();

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

  const handleSignOut = async () => {
    setSelectedPatient(null);
    await supabase.auth.signOut();
    navigate("/login");
  };

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
            {navItems
              .filter(item => {
                const passesAuth = !item.requiresAuth || session;
                const passesPatient = !item.requiresPatient || selectedPatient;
                return passesAuth && passesPatient;
              })
              .map((item) => (
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

          <div className="hidden md:flex items-center gap-2">
            {session && (
              <Link to="/admin">
                <Button
                  variant={location.pathname === "/admin" ? "default" : "ghost"}
                  size="sm"
                  className="gap-2"
                >
                  <ShieldCheck className="w-4 h-4" />
                  Admin
                </Button>
              </Link>
            )}
            {session ? (
              <Button variant="outline" size="sm" className="gap-2" onClick={handleSignOut}>
                <LogOut className="w-4 h-4" />
                Sign Out
              </Button>
            ) : (
              <Link to="/login">
                <Button variant="outline" size="sm" className="gap-2">
                  <LogIn className="w-4 h-4" />
                  Sign In
                </Button>
              </Link>
            )}
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
              {navItems
                .filter(item => {
                  const passesAuth = !item.requiresAuth || session;
                  const passesPatient = !item.requiresPatient || selectedPatient;
                  return passesAuth && passesPatient;
                })
                .map((item) => (
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
