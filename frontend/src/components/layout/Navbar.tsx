import { Link, useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Activity, Upload, Network, Menu, X, Users, FileSearch, LogIn, LogOut, ShieldCheck, Plus, Loader2, LayoutDashboard } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { usePatientContext } from "@/context/PatientContext";
import { useAuth } from "@/context/AuthContext";
import { useAnalysis } from "@/context/AnalysisContext";
import { toast } from "sonner";

const navItems = [
  { path: "/", label: "Home", icon: Activity, requiresAuth: false, requiresPatient: false, requiresReport: false },
  { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard, requiresAuth: true, requiresPatient: false, requiresReport: false },
  { path: "/patients", label: "Patients", icon: Users, requiresAuth: true, requiresPatient: false, requiresReport: false },
  { path: "/upload", label: "Upload", icon: Upload, requiresAuth: true, requiresPatient: true, requiresReport: false },
  { path: "/explainability", label: "Explainability", icon: FileSearch, requiresAuth: true, requiresPatient: true, requiresReport: true },
  { path: "/knowledge-graph", label: "Evidence Graph", icon: Network, requiresAuth: true, requiresPatient: true, requiresReport: true },
];

const AUTH_FEEDBACK_MS = 350;

export const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);
  const { session, isAdmin, signOut: authSignOut } = useAuth();
  const { selectedPatient, setSelectedPatient } = usePatientContext();
  const { report, currentScanId } = useAnalysis();
  const hasReport = !!report || !!currentScanId;
  const [signingOut, setSigningOut] = useState(false);

  const handleSignOut = async () => {
    setSigningOut(true);
    toast.dismiss();
    try {
      setSelectedPatient(null);
      await Promise.allSettled([
        authSignOut(),
        new Promise((resolve) => setTimeout(resolve, AUTH_FEEDBACK_MS)),
      ]);
    } finally {
      setIsOpen(false);
      setSigningOut(false);
      // Hard redirect to clear Supabase in-memory session cache completely
      window.location.href = "/login";
    }
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-xl">
      <div className="figma-container">
        <div className="flex items-center justify-between h-20">
          <Link to="/" className="flex items-center gap-2 group">
            <div className="relative w-10 h-10 rounded-xl bg-gradient-to-br from-sky-600 to-primary flex items-center justify-center shadow-sm transition-transform group-hover:scale-105">
              <Plus className="w-6 h-6 stroke-[4] text-white" />
            </div>
            <span className="block text-2xl font-extrabold tracking-tight text-foreground">
              XMedFusion
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-7">
            {navItems
              .filter(item => {
                const passesAuth = !item.requiresAuth || session;
                const passesPatient = !item.requiresPatient || selectedPatient;
                const passesReport = !item.requiresReport || hasReport;
                return passesAuth && passesPatient && passesReport;
              })
              .map((item) => (
                <Link key={item.path} to={item.path}>
                  <Button
                    variant="link"
                    size="sm"
                    className={cn(
                      "h-auto px-0 text-sm font-medium text-foreground/80 hover:text-primary",
                      location.pathname === item.path && "text-primary underline",
                    )}
                  >
                    {item.label}
                  </Button>
                </Link>
              ))}
          </div>

          <div className="hidden md:flex items-center gap-2">
            {session && isAdmin && (
              <Link to="/admin">
                <Button variant={location.pathname === "/admin" ? "default" : "ghost"} size="sm" className="gap-2 rounded-full">
                  <ShieldCheck className="w-4 h-4" />
                  Admin
                </Button>
              </Link>
            )}
            {session ? (
              <Button variant="outline" size="sm" className="gap-2 rounded-full" onClick={handleSignOut} disabled={signingOut}>
                {signingOut ? <Loader2 className="w-4 h-4 animate-spin" /> : <LogOut className="w-4 h-4" />}
                {signingOut ? "Signing out..." : "Sign Out"}
              </Button>
            ) : (
              <>
                <Link to="/login">
                  <Button variant="outline" size="sm" className="gap-2 rounded-full border-foreground/70 px-5 text-foreground hover:border-primary">
                    <LogIn className="w-4 h-4" />
                  Sign In
                  </Button>
                </Link>
                <Link to="/login">
                  <Button size="sm" className="rounded-full px-6">
                    Join As A Doctor
                  </Button>
                </Link>
              </>
            )}
          </div>

          <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setIsOpen(!isOpen)} aria-label="Toggle navigation">
            {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </Button>
        </div>

        {isOpen && (
          <div className="md:hidden py-4 animate-slide-up">
            <div className="flex flex-col gap-2">
              {navItems
                .filter(item => {
                  const passesAuth = !item.requiresAuth || session;
                  const passesPatient = !item.requiresPatient || selectedPatient;
                  const passesReport = !item.requiresReport || hasReport;
                  return passesAuth && passesPatient && passesReport;
                })
                .map((item) => (
                  <Link key={item.path} to={item.path} onClick={() => setIsOpen(false)}>
                    <Button variant={location.pathname === item.path ? "default" : "ghost"} className="w-full justify-start gap-2">
                      <item.icon className="w-4 h-4" />
                      {item.label}
                    </Button>
                  </Link>
                ))}
              {session && isAdmin && (
                <Link to="/admin" onClick={() => setIsOpen(false)}>
                  <Button variant={location.pathname === "/admin" ? "default" : "ghost"} className="w-full justify-start gap-2">
                    <ShieldCheck className="w-4 h-4" />
                    Admin
                  </Button>
                </Link>
              )}
              {session ? (
                <Button variant="outline" className="w-full justify-start gap-2" onClick={handleSignOut} disabled={signingOut}>
                  {signingOut ? <Loader2 className="w-4 h-4 animate-spin" /> : <LogOut className="w-4 h-4" />}
                  {signingOut ? "Signing out..." : "Sign Out"}
                </Button>
              ) : (
                <Link to="/login" onClick={() => setIsOpen(false)}>
                  <Button variant="outline" className="w-full justify-start gap-2">
                    <LogIn className="w-4 h-4" />
                    Sign In
                  </Button>
                </Link>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};
