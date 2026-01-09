import { useEffect } from "react";
import KnowledgeGraphComponent from "@/components/KnowledgeGraph";
import { useAnalysis } from "@/context/AnalysisContext";
import { Button } from "@/components/ui/button";
import { RefreshCcw } from "lucide-react";
import { Layout } from "@/components/layout/Layout"; // ADD LAYOUT HERE

const KnowledgeGraphPage = () => {
  const { knowledgeGraphData, resetAnalysis } = useAnalysis();

  const displayData = knowledgeGraphData || null;
  const isAnalysisActive = !!knowledgeGraphData;

  return (
    <Layout>
      <section className="py-12 lg:py-20">
        <div className="container mx-auto px-4">
          
          {/* Header Section - The ONLY header */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
            <div>
              <h1 className="text-3xl font-bold text-foreground mb-2">
                Knowledge Graph Visualization
              </h1>
              <p className="text-muted-foreground">
                {isAnalysisActive 
                  ? "Viewing live analysis results from your uploaded X-ray."
                  : "Viewing interactive demo mode (Static Data)."}
              </p>
            </div>

            {isAnalysisActive && (
              <Button 
                variant="outline" 
                onClick={resetAnalysis}
                className="flex items-center gap-2 text-muted-foreground hover:text-foreground"
              >
                <RefreshCcw className="w-4 h-4" />
                Reset to Demo
              </Button>
            )}
          </div>
          
          {/* Render Component */}
          <KnowledgeGraphComponent data={displayData} />

        </div>
      </section>
    </Layout>
  );
};

export default KnowledgeGraphPage;