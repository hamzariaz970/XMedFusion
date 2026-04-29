import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, Clock, LogOut, ShieldOff, RefreshCw } from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";

const PendingApproval = () => {
  const { isRejected, isApproved, isAdmin, session, signOut, refreshRole } = useAuth();
  const navigate = useNavigate();

  // Auto-redirect if user is already approved
  useEffect(() => {
    if (isApproved) {
      navigate(isAdmin ? "/admin" : "/upload", { replace: true });
    }
  }, [isApproved, isAdmin, navigate]);

  // Redirect to login if not logged in
  useEffect(() => {
    if (!session) {
      navigate("/login", { replace: true });
    }
  }, [session, navigate]);

  const handleSignOut = async () => {
    await signOut();
    navigate("/login");
  };

  const handleRefresh = async () => {
    await refreshRole();
    // The useEffect above will handle the redirect if approved
  };

  return (
    <div className="clinical-shell relative flex min-h-screen w-full items-center justify-center overflow-hidden px-4 py-12 sm:px-8 lg:px-14">
      <div className="w-full max-w-2xl space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
        {/* Logo */}
        <div className="text-center">
          <div className="inline-flex items-center gap-2 mb-6">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-glow">
              <Activity className="w-6 h-6" />
            </div>
            <span className="text-2xl font-bold tracking-tight text-foreground">
              XMed<span className="text-primary">Fusion</span>
            </span>
          </div>
        </div>

        <Card className="clinical-panel shadow-clinical">
          <CardContent className="pt-8 pb-8">
            <RadiologyImageCard
              src={radiologyImages.digitalConsult}
              alt="Clinical verification workflow"
              label="Secure review"
              caption="Credential approval required"
              className="mb-6 h-40"
              scanLine={false}
            />
            {isRejected ? (
              <div className="text-center space-y-4">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-destructive/15">
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
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/15">
                  <Clock className="w-8 h-8 text-amber-500 animate-pulse" />
                </div>
                <div className="mx-auto flex max-w-xs items-center justify-center gap-2 rounded-full border border-amber-500/20 bg-amber-500/10 px-3 py-1 text-xs font-semibold text-amber-600 dark:text-amber-400">
                  <span className="h-2 w-2 rounded-full bg-amber-500" />
                  Credential review in progress
                </div>
                <h2 className="text-2xl font-bold text-foreground">
                  Awaiting Approval
                </h2>
                <p className="text-muted-foreground text-sm leading-relaxed max-w-sm mx-auto">
                  Your registration is under review. A platform administrator
                  will verify your credentials and approve your account shortly.
                </p>
                <div className="rounded-[20px] border border-amber-500/20 bg-amber-500/10 px-4 py-3 pt-2">
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
