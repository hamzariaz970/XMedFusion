import React, { createContext, useContext, useState, ReactNode } from 'react';

// Define types
export interface ParsedReport {
  findings: string;
  impression: string;
  labels: string[];
  recommendation?: string;
}

export type FeedbackStatus = 'none' | 'reviewing' | 'draft' | 'approved';

interface AnalysisContextType {
  // Data
  uploadedFile: File | null;
  previewUrl: string | null;
  report: ParsedReport | null;
  knowledgeGraphData: any | null; 
  heatmapData: string | null;
  currentScanId: string | null;

  // HITL Feedback
  feedbackStatus: FeedbackStatus;
  setFeedbackStatus: (s: FeedbackStatus) => void;
  setCurrentScanId: (scanId: string | null) => void;
  
  // Actions
  setAnalysisResults: (
    file: File, 
    url: string, 
    report: ParsedReport, 
    kgData: any,
    heatmap: string | null,
    scanId?: string | null
  ) => void;
  
  resetAnalysis: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisProvider = ({ children }: { children: ReactNode }) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [report, setReport] = useState<ParsedReport | null>(null);
  const [knowledgeGraphData, setKgData] = useState<any | null>(null);
  const [heatmapData, setHeatmapData] = useState<string | null>(null);
  const [currentScanId, setCurrentScanId] = useState<string | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<FeedbackStatus>('none');

  const setAnalysisResults = (
    file: File, 
    url: string, 
    reportData: ParsedReport, 
    kgData: any,
    heatmap: string | null,
    scanId: string | null = null
  ) => {
    setUploadedFile(file);
    setPreviewUrl(url);
    setReport(reportData);
    setKgData(kgData);
    setHeatmapData(heatmap);
    setCurrentScanId(scanId);
    setFeedbackStatus('none');
  };

  const resetAnalysis = () => {
    setUploadedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl); 
    setPreviewUrl(null);
    setReport(null);
    setKgData(null);
    setHeatmapData(null);
    setCurrentScanId(null);
    setFeedbackStatus('none');
  };

  return (
    <AnalysisContext.Provider value={{ 
      uploadedFile, 
      previewUrl, 
      report, 
      knowledgeGraphData, 
      heatmapData,
      currentScanId,
      feedbackStatus,
      setFeedbackStatus,
      setCurrentScanId,
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
