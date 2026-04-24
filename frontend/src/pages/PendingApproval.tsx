import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, Clock, LogOut, ShieldOff, RefreshCw } from "lucide-react";
import { useAuth } from "@/context/AuthContext";

const PendingApproval = () => {
  const { isRejected, signOut, refreshRole } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await signOut();
    navigate("/login");
  };

  const handleRefresh = async () => {
    await refreshRole();
  };

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden bg-background py-12 px-4">
      {/* Background Orbs */}
      <div className="absolute inset-0 overflow-hidden -z-10">
        <div className="absolute top-1/4 -left-20 w-72 h-72 bg-primary/10 rounded-full blur-3xl animate-pulse-slow" />
        <div
          className="absolute bottom-1/4 -right-20 w-96 h-96 bg-medical-blue/10 rounded-full blur-3xl animate-pulse-slow"
          style={{ animationDelay: "2s" }}
        />
      </div>

      <div className="w-full max-w-md space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
        {/* Logo */}
        <div className="text-center">
          <div className="inline-flex items-center gap-2 mb-6">
            <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center text-primary-foreground shadow-glow">
              <Activity className="w-6 h-6" />
            </div>
            <span className="text-2xl font-bold tracking-tight text-foreground">
              XMed<span className="text-primary">Fusion</span>
            </span>
          </div>
        </div>

        <Card className="glass-card border-border/50 shadow-2xl">
          <CardContent className="pt-8 pb-8">
            {isRejected ? (
              <div className="text-center space-y-4">
                <div className="w-16 h-16 rounded-full bg-destructive/20 flex items-center justify-center mx-auto">
                  <ShieldOff className="w-8 h-8 text-destructive" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">
                  Registration Rejected
                </h2>
                <p className="text-muted-foreground text-sm leading-relaxed max-w-sm mx-auto">
                  Your registration request has been reviewed and was not
                  approved. Please contact the hospital administrator for more
                  information.
                </p>
              </div>
            ) : (
              <div className="text-center space-y-4">
                <div className="w-16 h-16 rounded-full bg-amber-500/20 flex items-center justify-center mx-auto">
                  <Clock className="w-8 h-8 text-amber-500 animate-pulse" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">
                  Awaiting Approval
                </h2>
                <p className="text-muted-foreground text-sm leading-relaxed max-w-sm mx-auto">
                  Your registration is under review. A platform administrator
                  will verify your credentials and approve your account shortly.
                </p>
                <div className="pt-2 px-4 py-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
                  <p className="text-xs text-amber-600 dark:text-amber-400">
                    You will be able to access the platform once your account is
                    approved. Please check back later.
                  </p>
                </div>
              </div>
            )}

            <div className="flex flex-col gap-3 mt-6">
              <Button
                variant="outline"
                className="w-full gap-2"
                onClick={handleRefresh}
              >
                <RefreshCw className="w-4 h-4" />
                Check Status
              </Button>
              <Button
                variant="ghost"
                className="w-full gap-2 text-muted-foreground"
                onClick={handleSignOut}
              >
                <LogOut className="w-4 h-4" />
                Sign Out
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PendingApproval;
