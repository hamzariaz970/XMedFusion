import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { Activity } from "lucide-react";

interface ProtectedRouteProps {
  requireAdmin?: boolean;
}

export const ProtectedRoute = ({ requireAdmin = false }: ProtectedRouteProps) => {
  const { session, loading, roleLoading, isAdmin, isPending, isRejected, isApproved, userRole, isRestoringSession } = useAuth();

  // Hold the spinner while the initial auth state OR the user_roles row is loading.
  // Without the roleLoading check, a hard reload causes a brief window where
  // loading=false, session!=null, but userRole=null → spurious redirect to /pending.
  if (loading || roleLoading) {
    const statusMessage = isRestoringSession
      ? "Verifying your session and access..."
      : "Loading...";
    const detailMessage = isRestoringSession
      ? "This security check can take a few seconds after a refresh."
      : null;

    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4 animate-in fade-in duration-500">
          <div className="w-12 h-12 rounded-xl bg-primary flex items-center justify-center shadow-glow">
            <Activity className="w-6 h-6 text-primary-foreground animate-pulse" />
          </div>
          <div className="space-y-1 text-center">
            <p className="text-sm font-medium text-foreground">{statusMessage}</p>
            {detailMessage ? (
              <p className="max-w-xs text-xs text-muted-foreground">{detailMessage}</p>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  // Not logged in at all
  if (!session) {
    return <Navigate to="/login" replace />;
  }

  // User has no role entry yet (shouldn't happen in normal flow, but safety net)
  // Or user is pending/rejected => show pending page
  if (!userRole || isPending || isRejected) {
    return <Navigate to="/pending" replace />;
  }

  // Route requires admin but user is not admin
  if (requireAdmin && !isAdmin) {
    return <Navigate to="/dashboard" replace />;
  }

  // Doctor must be approved to access protected routes
  if (!isApproved) {
    return <Navigate to="/pending" replace />;
  }

  return <Outlet />;
};
