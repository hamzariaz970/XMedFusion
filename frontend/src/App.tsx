import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import PatientDashboard from "./pages/PatientDashboard";
import UploadXray from "./pages/UploadXray";
import ExplainabilityModule from "./pages/ExplainabilityModule";
import KnowledgeGraph from "./pages/KnowledgeGraph";
import ImageMapping from "./pages/ImageMapping";
import NotFound from "./pages/NotFound";
import { AnalysisProvider } from "@/context/AnalysisContext"; // Import the provider

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AnalysisProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/patients" element={<PatientDashboard />} />
            <Route path="/upload" element={<UploadXray />} />
            <Route path="/explainability" element={<ExplainabilityModule />} />
            <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
            <Route path="/image-mapping" element={<ImageMapping />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </AnalysisProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
