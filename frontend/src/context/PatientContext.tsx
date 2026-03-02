import React, { createContext, useContext, useState, useEffect } from 'react';
import { supabase } from '@/lib/supabaseClient';

export interface Patient {
    id: string;
    name: string;
    age: number;
    gender: string;
    conditions: string[];
    notes?: string;
    status: string;
    user_id: string;
    created_at: string;
    updated_at: string;
}

interface PatientContextType {
    selectedPatient: Patient | null;
    setSelectedPatient: (patient: Patient | null) => void;
    patients: Patient[];
    loading: boolean;
    refreshPatients: () => Promise<void>;
    pendingUploadFiles: File[] | null;
    setPendingUploadFiles: (files: File[] | null) => void;
    pendingScanType: string;
    setPendingScanType: (type: string) => void;
}

const PatientContext = createContext<PatientContextType | undefined>(undefined);

export const PatientProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
    const [patients, setPatients] = useState<Patient[]>([]);
    const [loading, setLoading] = useState(true);
    const [pendingUploadFiles, setPendingUploadFiles] = useState<File[] | null>(null);
    const [pendingScanType, setPendingScanType] = useState<string>('auto');

    const fetchPatients = async () => {
        try {
            setLoading(true);
            const { data: { session } } = await supabase.auth.getSession();

            if (!session) {
                setPatients([]);
                setSelectedPatient(null);
                return;
            }

            const { data, error } = await supabase
                .from('patients')
                .select('*')
                .order('created_at', { ascending: false });

            if (error) {
                console.error('Error fetching patients:', error);
                return;
            }

            setPatients(data || []);

            // If we have a selected patient, check if they still exist in the fetched list
            if (selectedPatient) {
                const stillExists = data?.find(p => p.id === selectedPatient.id);
                if (!stillExists) {
                    setSelectedPatient(null);
                } else {
                    // Update selected patient data in case it changed
                    setSelectedPatient(stillExists);
                }
            }
        } catch (err) {
            console.error('Unexpected error fetching patients:', err);
        } finally {
            setLoading(false);
        }
    };

    // Fetch patients on mount and when auth state changes
    useEffect(() => {
        fetchPatients();

        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            if (session) {
                fetchPatients();
            } else {
                setPatients([]);
                setSelectedPatient(null);
            }
        });

        return () => {
            subscription.unsubscribe();
        };
    }, []);

    return (
        <PatientContext.Provider
            value={{
                selectedPatient,
                setSelectedPatient,
                patients,
                loading,
                refreshPatients: fetchPatients,
                pendingUploadFiles,
                setPendingUploadFiles,
                pendingScanType,
                setPendingScanType,
            }}
        >
            {children}
        </PatientContext.Provider>
    );
};

export const usePatientContext = () => {
    const context = useContext(PatientContext);
    if (context === undefined) {
        throw new Error('usePatientContext must be used within a PatientProvider');
    }
    return context;
};
