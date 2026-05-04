import React, { createContext, ReactNode, useContext, useEffect, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../lib/supabase';

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
}

const LAST_PATIENT_STORAGE_KEY = 'xmedfusion:last-patient-id';
const PatientContext = createContext<PatientContextType | undefined>(undefined);

export function PatientProvider({ children }: { children: ReactNode }) {
  const [selectedPatient, setSelectedPatientState] = useState<Patient | null>(null);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);

  const setSelectedPatient = (patient: Patient | null) => {
    setSelectedPatientState(patient);
    if (patient) {
      void AsyncStorage.setItem(LAST_PATIENT_STORAGE_KEY, patient.id);
    } else {
      void AsyncStorage.removeItem(LAST_PATIENT_STORAGE_KEY);
    }
  };

  const refreshPatients = async () => {
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
        console.error('Error fetching patients:', error.message);
        return;
      }

      const nextPatients = (data || []) as Patient[];
      setPatients(nextPatients);

      if (selectedPatient) {
        const refreshedSelected = nextPatients.find((patient) => patient.id === selectedPatient.id) || null;
        setSelectedPatient(refreshedSelected);
      } else {
        const lastPatientId = await AsyncStorage.getItem(LAST_PATIENT_STORAGE_KEY);
        const lastPatient = nextPatients.find((patient) => patient.id === lastPatientId) || null;
        if (lastPatient) {
          setSelectedPatientState(lastPatient);
        }
      }
    } catch (error) {
      console.error('Unexpected error fetching patients:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refreshPatients();

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session) {
        void refreshPatients();
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
        refreshPatients,
      }}
    >
      {children}
    </PatientContext.Provider>
  );
}

export const usePatientContext = () => {
  const context = useContext(PatientContext);
  if (!context) {
    throw new Error('usePatientContext must be used within a PatientProvider');
  }
  return context;
};
