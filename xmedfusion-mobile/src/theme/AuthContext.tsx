import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { Session, User } from '@supabase/supabase-js';
import { supabase } from '../lib/supabase';

type AuthContextType = {
  session: Session | null;
  user: User | null;
  loading: boolean;
  roleLoading: boolean;
  userRole: string | null;
  isAdmin: boolean;
  isApproved: boolean;
  isPending: boolean;
  isRejected: boolean;
  refreshRole: () => Promise<void>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType>({
  session: null,
  user: null,
  loading: true,
  roleLoading: true,
  userRole: null,
  isAdmin: false,
  isApproved: false,
  isPending: false,
  isRejected: false,
  refreshRole: async () => {},
  signOut: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [roleLoading, setRoleLoading] = useState(true);
  const [userRole, setUserRole] = useState<string | null>(null);
  const [approvalStatus, setApprovalStatus] = useState<'pending' | 'approved' | 'rejected' | null>(null);

  useEffect(() => {
    // Check for active session
    console.log('[Supabase] Initializing session check...');
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (error) {
        console.error('[Supabase] Connection error:', error.message);
      } else {
        console.log('[Supabase] Session initialized:', session ? 'Active' : 'No active session');
      }
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
    });

    // Listen for auth changes
    console.log('[Supabase] Setting up auth change listener...');
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);
      }
    );

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  const refreshRole = async () => {
    if (!session?.user) {
      setUserRole(null);
      setApprovalStatus(null);
      setRoleLoading(false);
      return;
    }

    setRoleLoading(true);
    const { data, error } = await supabase
      .from('user_roles')
      .select('role, approval_status')
      .eq('user_id', session.user.id)
      .maybeSingle();

    if (error) {
      console.error('[Supabase] Failed to load role:', error.message);
      setUserRole(null);
      setApprovalStatus(null);
    } else {
      setUserRole(data?.role ?? null);
      setApprovalStatus((data?.approval_status as 'pending' | 'approved' | 'rejected' | null) ?? null);
    }

    setRoleLoading(false);
  };

  useEffect(() => {
    let cancelled = false;

    const loadRole = async () => {
      if (!session?.user) {
        setUserRole(null);
        setApprovalStatus(null);
        setRoleLoading(false);
        return;
      }

      setRoleLoading(true);
      const { data, error } = await supabase
        .from('user_roles')
        .select('role, approval_status')
        .eq('user_id', session.user.id)
        .maybeSingle();

      if (cancelled) return;

      if (error) {
        console.error('[Supabase] Failed to load role:', error.message);
        setUserRole(null);
        setApprovalStatus(null);
      } else {
        setUserRole(data?.role ?? null);
        setApprovalStatus((data?.approval_status as 'pending' | 'approved' | 'rejected' | null) ?? null);
      }

      setRoleLoading(false);
    };

    loadRole();

    return () => {
      cancelled = true;
    };
  }, [session?.user?.id]);

  const isAdmin = userRole === 'admin';
  const isApproved = approvalStatus === 'approved';
  const isPending = approvalStatus === 'pending' || !approvalStatus;
  const isRejected = approvalStatus === 'rejected';

  const signOut = async () => {
    await supabase.auth.signOut();
  };

  return (
    <AuthContext.Provider value={{
      session,
      user,
      loading,
      roleLoading,
      userRole,
      isAdmin,
    isApproved,
    isPending,
    isRejected,
    refreshRole,
    signOut,
  }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
