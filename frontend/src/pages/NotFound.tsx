import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { RadiologyImageCard } from "@/components/RadiologyImageCard";
import { radiologyImages } from "@/assets/radiology";
import { Activity, ArrowLeft, FileSearch } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useEffect } from "react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <Layout>
      <div className="figma-container flex min-h-[70vh] items-center justify-center py-16">
        <div className="clinical-panel max-w-xl p-8 text-center shadow-clinical">
          <RadiologyImageCard
            src={radiologyImages.neuroReview}
            alt="Radiology workspace search"
            className="mb-6 h-48"
            label="Workspace search"
            caption="Route not found"
            scanLine={false}
          />
          <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <FileSearch className="h-8 w-8" />
          </div>
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">404 diagnostic miss</p>
          <h1 className="mt-3 text-4xl font-extrabold text-foreground">Page not <span className="text-primary">found</span></h1>
          <p className="mx-auto mt-4 max-w-md text-sm leading-6 text-muted-foreground">
            The requested route is not part of the current XMedFusion clinical workspace.
          </p>
          <div className="mt-8 flex flex-col justify-center gap-3 sm:flex-row">
            <Link to="/">
              <Button variant="hero" className="w-full sm:w-auto">
                <ArrowLeft className="h-4 w-4" />
                Return Home
              </Button>
            </Link>
            <Link to="/knowledge-graph">
              <Button variant="outline" className="w-full sm:w-auto">
                <Activity className="h-4 w-4" />
                Open Graph
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default NotFound;
