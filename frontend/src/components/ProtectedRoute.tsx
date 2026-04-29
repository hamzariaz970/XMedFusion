import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { Activity } from "lucide-react";

interface ProtectedRouteProps {
  requireAdmin?: boolean;
}

export const ProtectedRoute = ({ requireAdmin = false }: ProtectedRouteProps) => {
  const { session, loading, isAdmin, isPending, isRejected, isApproved, userRole } = useAuth();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4 animate-in fade-in duration-500">
          <div className="w-12 h-12 rounded-xl bg-primary flex items-center justify-center shadow-glow">
            <Activity className="w-6 h-6 text-primary-foreground animate-pulse" />
          </div>
          <p className="text-sm text-muted-foreground">Loading...</p>
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
