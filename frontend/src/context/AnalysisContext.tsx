// @refresh reset
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
  referenceImageUrl: string | null;
  report: ParsedReport | null;
  knowledgeGraphData: any | null; 
  heatmapData: string | null;
  detectedModality: string | null;
  explainabilityData: any | null;
  currentScanId: string | null;

  // HITL Feedback
  feedbackStatus: FeedbackStatus;
  setFeedbackStatus: (s: FeedbackStatus) => void;
  setCurrentScanId: (scanId: string | null) => void;
  updateReport: (reportPatch: Partial<ParsedReport>) => void;
  
  // Actions
  setAnalysisResults: (
    file: File, 
    url: string, 
    report: ParsedReport, 
    kgData: any,
    heatmap: string | null,
    detectedModality?: string | null,
    explainabilityData?: any | null,
    referenceImageUrl?: string | null,
    scanId?: string | null
  ) => void;
  
  resetAnalysis: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisProvider = ({ children }: { children: ReactNode }) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [referenceImageUrl, setReferenceImageUrl] = useState<string | null>(null);
  const [report, setReport] = useState<ParsedReport | null>(null);
  const [knowledgeGraphData, setKgData] = useState<any | null>(null);
  const [heatmapData, setHeatmapData] = useState<string | null>(null);
  const [detectedModality, setDetectedModality] = useState<string | null>(null);
  const [explainabilityData, setExplainabilityData] = useState<any | null>(null);
  const [currentScanId, setCurrentScanId] = useState<string | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<FeedbackStatus>('none');

  const updateReport = (reportPatch: Partial<ParsedReport>) => {
    setReport((prev) => (prev ? { ...prev, ...reportPatch } : prev));
  };

  const setAnalysisResults = (
    file: File, 
    url: string, 
    reportData: ParsedReport, 
    kgData: any,
    heatmap: string | null,
    modality: string | null = null,
    explainabilityPayload: any | null = null,
    referenceUrl: string | null = null,
    scanId: string | null = null
  ) => {
    setUploadedFile(file);
    setPreviewUrl(url);
    setReferenceImageUrl(referenceUrl);
    setReport(reportData);
    setKgData(kgData);
    setHeatmapData(heatmap);
    setDetectedModality(modality);
    setExplainabilityData(explainabilityPayload);
    setCurrentScanId(scanId);
    setFeedbackStatus('none');
  };

  const resetAnalysis = () => {
    setUploadedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl); 
    setPreviewUrl(null);
    setReferenceImageUrl(null);
    setReport(null);
    setKgData(null);
    setHeatmapData(null);
    setDetectedModality(null);
    setExplainabilityData(null);
    setCurrentScanId(null);
    setFeedbackStatus('none');
  };

  return (
    <AnalysisContext.Provider value={{ 
      uploadedFile, 
      previewUrl, 
      referenceImageUrl,
      report, 
      knowledgeGraphData, 
      heatmapData,
      detectedModality,
      explainabilityData,
      currentScanId,
      feedbackStatus,
      setFeedbackStatus,
      setCurrentScanId,
      updateReport,
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
