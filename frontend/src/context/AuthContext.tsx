import { createContext, useContext, useEffect, useState, useCallback, ReactNode } from "react";
import { clearSupabaseAuthStorage, supabase } from "@/lib/supabaseClient";
import type { Session, User } from "@supabase/supabase-js";

// ---------- Types ----------
export interface UserRole {
  id: string;
  user_id: string;
  role: "admin" | "doctor";
  approval_status: "pending" | "approved" | "rejected";
}

interface AuthContextValue {
  session: Session | null;
  user: User | null;
  userRole: UserRole | null;
  isAdmin: boolean;
  isDoctor: boolean;
  isPending: boolean;
  isRejected: boolean;
  isApproved: boolean;
  loading: boolean;
  refreshRole: () => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);
const AUTH_TIMEOUT_MS = 8000;
const SIGN_OUT_TIMEOUT_MS = 1500;

const withTimeout = async <T,>(promise: Promise<T>, fallback: T, timeoutMs = AUTH_TIMEOUT_MS): Promise<T> => {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<T>((resolve) => {
    timeoutId = setTimeout(() => resolve(fallback), timeoutMs);
  });

  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
};

// ---------- Provider ----------
export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [userRole, setUserRole] = useState<UserRole | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchRole = useCallback(async (userId: string) => {
    const { data, error } = await supabase
      .from("user_roles")
      .select("*")
      .eq("user_id", userId)
      .maybeSingle();
    if (!error && data) {
      setUserRole(data as UserRole);
    } else {
      setUserRole(null);
    }
  }, []);

  const refreshRole = useCallback(async () => {
    if (user) {
      await fetchRole(user.id);
    }
  }, [user, fetchRole]);

  useEffect(() => {
    let mounted = true;

    // Initial session
    withTimeout(supabase.auth.getSession(), { data: { session: null }, error: null }).then(async ({ data: { session } }) => {
      if (!mounted) return;
      setSession(session);
      setUser(session?.user ?? null);
      if (session?.user) {
        await withTimeout(fetchRole(session.user.id), undefined);
      }
      if (!mounted) return;
      setLoading(false);
    }).catch((error) => {
      console.warn("Auth session initialization warning:", error);
      if (!mounted) return;
      setSession(null);
      setUser(null);
      setUserRole(null);
      setLoading(false);
    });

    // Auth state changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (_event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
      if (session?.user) {
        await fetchRole(session.user.id);
      } else {
        setUserRole(null);
      }
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, [fetchRole]);

  const signOut = useCallback(async () => {
    setSession(null);
    setUser(null);
    setUserRole(null);

    try {
      const { error } = await withTimeout(
        supabase.auth.signOut(),
        { error: new Error("Sign-out cleanup timed out.") },
        SIGN_OUT_TIMEOUT_MS
      );
      if (error) {
        console.warn("Supabase sign-out warning:", error.message);
      }
    } finally {
      setSession(null);
      setUser(null);
      setUserRole(null);
      clearSupabaseAuthStorage();
    }
  }, []);

  const isAdmin = userRole?.role === "admin" && userRole?.approval_status === "approved";
  const isDoctor = userRole?.role === "doctor";
  const isPending = userRole?.approval_status === "pending";
  const isRejected = userRole?.approval_status === "rejected";
  const isApproved = userRole?.approval_status === "approved";

  return (
    <AuthContext.Provider
      value={{
        session,
        user,
        userRole,
        isAdmin,
        isDoctor,
        isPending,
        isRejected,
        isApproved,
        loading,
        refreshRole,
        signOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

// ---------- Hook ----------
export const useAuth = (): AuthContextValue => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
};
