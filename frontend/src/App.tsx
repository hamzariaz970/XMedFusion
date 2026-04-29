import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { useEffect } from "react";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { AuthProvider } from "@/context/AuthContext";
import { AnalysisProvider } from "@/context/AnalysisContext";
import { PatientProvider } from "@/context/PatientContext";
import Index from "./pages/Index";
import Login from "./pages/Login";
import PendingApproval from "./pages/PendingApproval";
import AdminDashboard from "./pages/AdminDashboard";
import DoctorDashboard from "./pages/DoctorDashboard";
import PatientDashboard from "./pages/PatientDashboard";
import UploadXray from "./pages/UploadXray";
import ExplainabilityModule from "./pages/ExplainabilityModule";
import ImageMapping from "./pages/ImageMapping";
import KnowledgeGraphPage from "./pages/KnowledgeGraphPage";
import NotFound from "./pages/NotFound";
import HILLabelingPage from "./pages/HILLabelingPage";

const queryClient = new QueryClient();

const ScrollToTop = () => {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, [pathname]);

  return null;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <AnalysisProvider>
          <PatientProvider>
            <Toaster />
            <Sonner />
            <BrowserRouter>
              <ScrollToTop />
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/login" element={<Login />} />
                <Route path="/pending" element={<PendingApproval />} />
                <Route path="/knowledge-graph" element={<KnowledgeGraphPage />} />

                {/* Admin-only route */}
                <Route element={<ProtectedRoute requireAdmin />}>
                  <Route path="/admin" element={<AdminDashboard />} />
                </Route>

                {/* Doctor/Admin routes */}
                <Route element={<ProtectedRoute />}>
                  <Route path="/dashboard" element={<DoctorDashboard />} />
                  <Route path="/patients" element={<PatientDashboard />} />
                  <Route path="/upload" element={<UploadXray />} />
                  <Route path="/explainability" element={<ExplainabilityModule />} />
                  <Route path="/image-mapping" element={<ImageMapping />} />
                  <Route path="/hil/task/:taskId" element={<HILLabelingPage />} />
                </Route>

                <Route path="*" element={<NotFound />} />
              </Routes>
            </BrowserRouter>
          </PatientProvider>
        </AnalysisProvider>
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
