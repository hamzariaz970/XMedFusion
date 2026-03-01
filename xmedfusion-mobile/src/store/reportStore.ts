import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface ReportItem {
    id: string;
    patientId: string;
    date: string;
    findings: string;
    impression: string;
    knowledgeGraph: any;
}

interface ReportState {
    reports: ReportItem[];
    addReport: (report: ReportItem) => void;
    clearReports: () => void;
}

export const useReportStore = create<ReportState>()(
    persist(
        (set) => ({
            reports: [],
            addReport: (report) => set((state) => ({ reports: [report, ...state.reports] })),
            clearReports: () => set({ reports: [] }),
        }),
        {
            name: 'report-storage',
            storage: createJSONStorage(() => AsyncStorage),
        }
    )
);
