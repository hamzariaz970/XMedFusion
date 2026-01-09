import React, { createContext, useContext, useState, ReactNode } from 'react';

// Define types
export interface ParsedReport {
  findings: string;
  impression: string;
  labels: string[];
}

interface AnalysisContextType {
  // Data
  uploadedFile: File | null;
  previewUrl: string | null;
  report: ParsedReport | null;
  knowledgeGraphData: any | null; 
  heatmapData: string | null; // <--- NEW FIELD (Base64 Data URI)
  
  // Actions
  setAnalysisResults: (
    file: File, 
    url: string, 
    report: ParsedReport, 
    kgData: any,
    heatmap: string | null // <--- UPDATED SIGNATURE
  ) => void;
  
  resetAnalysis: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisProvider = ({ children }: { children: ReactNode }) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [report, setReport] = useState<ParsedReport | null>(null);
  const [knowledgeGraphData, setKgData] = useState<any | null>(null);
  const [heatmapData, setHeatmapData] = useState<string | null>(null); // <--- NEW STATE

  const setAnalysisResults = (
    file: File, 
    url: string, 
    reportData: ParsedReport, 
    kgData: any,
    heatmap: string | null // <--- RECEIVE HEATMAP
  ) => {
    setUploadedFile(file);
    setPreviewUrl(url);
    setReport(reportData);
    setKgData(kgData);
    setHeatmapData(heatmap); // <--- SAVE IT
  };

  const resetAnalysis = () => {
    setUploadedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl); 
    setPreviewUrl(null);
    setReport(null);
    setKgData(null);
    setHeatmapData(null); // <--- RESET IT
  };

  return (
    <AnalysisContext.Provider value={{ 
      uploadedFile, 
      previewUrl, 
      report, 
      knowledgeGraphData, 
      heatmapData, // <--- EXPOSE IT
      setAnalysisResults, 
      resetAnalysis 
    }}>
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (context === undefined) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};