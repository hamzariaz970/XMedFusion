import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { AuthProvider } from "@/context/AuthContext";
import { AnalysisProvider } from "@/context/AnalysisContext";
import { PatientProvider } from "@/context/PatientContext";
import Index from "./pages/Index";
import Login from "./pages/Login";
import PendingApproval from "./pages/PendingApproval";
import AdminDashboard from "./pages/AdminDashboard";
import PatientDashboard from "./pages/PatientDashboard";
import UploadXray from "./pages/UploadXray";
import ExplainabilityModule from "./pages/ExplainabilityModule";
import ImageMapping from "./pages/ImageMapping";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <AnalysisProvider>
          <PatientProvider>
            <Toaster />
            <Sonner />
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/login" element={<Login />} />
                <Route path="/pending" element={<PendingApproval />} />

                {/* Admin-only route */}
                <Route element={<ProtectedRoute requireAdmin />}>
                  <Route path="/admin" element={<AdminDashboard />} />
                </Route>

                {/* Doctor/Admin routes */}
                <Route element={<ProtectedRoute />}>
                  <Route path="/patients" element={<PatientDashboard />} />
                  <Route path="/upload" element={<UploadXray />} />
                  <Route path="/explainability" element={<ExplainabilityModule />} />
                  <Route path="/image-mapping" element={<ImageMapping />} />
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
