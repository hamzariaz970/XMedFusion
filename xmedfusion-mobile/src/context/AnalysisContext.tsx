import React, { createContext, ReactNode, useContext, useState } from 'react';

export interface ParsedReport {
  findings: string;
  impression: string;
  labels: string[];
  recommendation?: string;
}

interface AnalysisContextType {
  previewUrl: string | null;
  previewUrls: string[];
  referenceImageUrl: string | null;
  report: ParsedReport | null;
  knowledgeGraphData: any | null;
  heatmapData: string | null;
  detectedModality: string | null;
  explainabilityData: any | null;
  currentScanId: string | null;
  setCurrentScanId: (scanId: string | null) => void;
  updateReport: (reportPatch: Partial<ParsedReport>) => void;
  setAnalysisResults: (
    url: string,
    report: ParsedReport,
    kgData: any,
    heatmap: string | null,
    detectedModality?: string | null,
    explainabilityData?: any | null,
    referenceImageUrl?: string | null,
    scanId?: string | null,
    allUrls?: string[]
  ) => void;
  resetAnalysis: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [referenceImageUrl, setReferenceImageUrl] = useState<string | null>(null);
  const [report, setReport] = useState<ParsedReport | null>(null);
  const [knowledgeGraphData, setKnowledgeGraphData] = useState<any | null>(null);
  const [heatmapData, setHeatmapData] = useState<string | null>(null);
  const [detectedModality, setDetectedModality] = useState<string | null>(null);
  const [explainabilityData, setExplainabilityData] = useState<any | null>(null);
  const [currentScanId, setCurrentScanId] = useState<string | null>(null);

  const updateReport = (reportPatch: Partial<ParsedReport>) => {
    setReport((previous) => (previous ? { ...previous, ...reportPatch } : previous));
  };

  const setAnalysisResults = (
    url: string,
    reportData: ParsedReport,
    kgData: any,
    heatmap: string | null,
    modality: string | null = null,
    explainabilityPayload: any | null = null,
    referenceUrl: string | null = null,
    scanId: string | null = null,
    allUrls: string[] = []
  ) => {
    setPreviewUrl(url);
    setPreviewUrls(allUrls.length > 0 ? allUrls : [url]);
    setReferenceImageUrl(referenceUrl);
    setReport(reportData);
    setKnowledgeGraphData(kgData);
    setHeatmapData(heatmap);
    setDetectedModality(modality);
    setExplainabilityData(explainabilityPayload);
    setCurrentScanId(scanId);
  };

  const resetAnalysis = () => {
    setPreviewUrl(null);
    setPreviewUrls([]);
    setReferenceImageUrl(null);
    setReport(null);
    setKnowledgeGraphData(null);
    setHeatmapData(null);
    setDetectedModality(null);
    setExplainabilityData(null);
    setCurrentScanId(null);
  };

  return (
    <AnalysisContext.Provider
      value={{
        previewUrl,
        previewUrls,
        referenceImageUrl,
        report,
        knowledgeGraphData,
        heatmapData,
        detectedModality,
        explainabilityData,
        currentScanId,
        setCurrentScanId,
        updateReport,
        setAnalysisResults,
        resetAnalysis,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
}

export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};
