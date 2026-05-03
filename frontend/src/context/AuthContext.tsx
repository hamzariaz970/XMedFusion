// @refresh reset
import { createContext, useContext, useEffect, useState, useCallback, ReactNode, useRef } from "react";
import { clearSupabaseAuthStorage, hasPersistedSupabaseSession, supabase } from "@/lib/supabaseClient";
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
  /** True while the user_roles row is being fetched from Supabase. ProtectedRoute
   *  must wait for this to be false before making any routing decisions. */
  roleLoading: boolean;
  isRestoringSession: boolean;
  refreshRole: () => Promise<void>;
  signOut: () => Promise<void>;
}

interface RoleFetchResult {
  data: UserRole | null;
  error: { message?: string } | null;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);
const AUTH_TIMEOUT_MS = 8000;
const USER_ROLE_STORAGE_KEY = "xmedfusion-user-role";
// Give the server-side token invalidation call enough time to complete.
// 1500ms was too short — the refresh token would stay alive on Supabase's
// servers even though localStorage was cleared.
const SIGN_OUT_TIMEOUT_MS = 10000;

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
  const [roleLoading, setRoleLoading] = useState(false);
  const [isRestoringSession, setIsRestoringSession] = useState(() => hasPersistedSupabaseSession());
  const userRef = useRef<User | null>(null);
  const userRoleRef = useRef<UserRole | null>(null);
  const roleRequestIdRef = useRef(0);

  // Role fetch with a timeout so a hanging REST call (e.g. after tab wake-up
  // when the browser's network stack is reconnecting) can't block forever.
  const ROLE_FETCH_TIMEOUT_MS = 10000;
  const ROLE_FETCH_TIMEOUT = "__role_fetch_timeout__";

  const readCachedUserRole = useCallback((userId: string): UserRole | null => {
    if (typeof window === "undefined") return null;

    try {
      const raw = window.localStorage.getItem(USER_ROLE_STORAGE_KEY);
      if (!raw) return null;

      const parsed = JSON.parse(raw) as UserRole;
      return parsed.user_id === userId ? parsed : null;
    } catch {
      return null;
    }
  }, []);

  const writeCachedUserRole = useCallback((nextRole: UserRole | null) => {
    if (typeof window === "undefined") return;

    if (!nextRole) {
      window.localStorage.removeItem(USER_ROLE_STORAGE_KEY);
      return;
    }

    window.localStorage.setItem(USER_ROLE_STORAGE_KEY, JSON.stringify(nextRole));
  }, []);

  const setTrackedUser = useCallback((nextUser: User | null) => {
    userRef.current = nextUser;
    setUser(nextUser);
  }, []);

  const setTrackedUserRole = useCallback((nextRole: UserRole | null) => {
    userRoleRef.current = nextRole;
    setUserRole(nextRole);
    writeCachedUserRole(nextRole);
  }, [writeCachedUserRole]);

  const fetchRole = useCallback(async (
    userId: string,
    options?: { block?: boolean; preserveOnError?: boolean }
  ) => {
    const block = options?.block ?? true;
    const preserveOnError = options?.preserveOnError ?? false;
    const requestId = ++roleRequestIdRef.current;

    if (block) {
      setRoleLoading(true);
    }

    try {
      const rolePromise = supabase
        .from("user_roles")
        .select("*")
        .eq("user_id", userId)
        .maybeSingle()
        .then(({ data, error }) => ({
          data: (data as UserRole | null) ?? null,
          error: error ? { message: error.message } : null,
        } satisfies RoleFetchResult));

      const result = await withTimeout<RoleFetchResult | typeof ROLE_FETCH_TIMEOUT>(
        rolePromise,
        ROLE_FETCH_TIMEOUT,
        ROLE_FETCH_TIMEOUT_MS
      );

      if (result === ROLE_FETCH_TIMEOUT) {
        if (requestId !== roleRequestIdRef.current || userRef.current?.id !== userId) {
          return;
        }

        console.warn("Auth: role fetch timed out.");
        if (!preserveOnError) {
          setTrackedUserRole(null);
        }
        return;
      }

      const { data, error } = result;

      if (requestId !== roleRequestIdRef.current || userRef.current?.id !== userId) {
        return;
      }

      if (error) {
        console.warn("Auth: failed to fetch user role.", error.message);
        if (!preserveOnError) {
          setTrackedUserRole(null);
        }
        return;
      }

      setTrackedUserRole((data ?? null) as UserRole | null);
    } finally {
      if (block && requestId === roleRequestIdRef.current && userRef.current?.id === userId) {
        setRoleLoading(false);
      }
    }
  }, [setTrackedUserRole]);

  const refreshRole = useCallback(async () => {
    if (userRef.current) {
      await fetchRole(userRef.current.id, {
        block: !userRoleRef.current,
        preserveOnError: !!userRoleRef.current,
      });
    }
  }, [fetchRole]);

  const reconcileSessionRole = useCallback(async (
    nextUser: User | null,
    optimisticRole: UserRole | null,
    needsBlockingRoleFetch: boolean
  ) => {
    if (nextUser) {
      if (optimisticRole) {
        setTrackedUserRole(optimisticRole);
      }

      await fetchRole(nextUser.id, {
        block: needsBlockingRoleFetch,
        preserveOnError: !!optimisticRole,
      });
      return;
    }

    roleRequestIdRef.current += 1;
    setTrackedUserRole(null);
    setRoleLoading(false);
  }, [fetchRole, setTrackedUserRole]);

  useEffect(() => {
    let mounted = true;
    let initialised = false;

    // Use onAuthStateChange as the SOLE session initializer.
    // Supabase v2 fires INITIAL_SESSION synchronously during subscription
    // setup, so this handles both initial load and subsequent changes
    // (TOKEN_REFRESHED, SIGNED_OUT, etc.) without any race conditions.
    //
    // Previously we also called getSession().then(fetchRole) which raced
    // with the INITIAL_SESSION callback — if getSession's 8s timeout fired
    // first, its {session:null} fallback overwrote the valid session that
    // onAuthStateChange had already set.
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!mounted) return;

      const nextUser = session?.user ?? null;
      const previousUserId = userRef.current?.id;
      const previousRole = userRoleRef.current;
      const cachedRole = nextUser ? readCachedUserRole(nextUser.id) : null;
      const optimisticRole = previousUserId === nextUser?.id ? previousRole : cachedRole;
      const needsBlockingRoleFetch = !!nextUser && !optimisticRole;

      setSession(session);
      setTrackedUser(nextUser);

      if (needsBlockingRoleFetch) {
        setRoleLoading(true);
      }

      if (!nextUser) {
        setRoleLoading(false);
      }

      // Mark initialization complete on the first event (INITIAL_SESSION).
      if (!initialised) {
        initialised = true;
        setLoading(false);
        setIsRestoringSession(false);
      }

      setTimeout(() => {
        if (!mounted) return;
        void reconcileSessionRole(nextUser, optimisticRole, needsBlockingRoleFetch);
      }, 0);
    });

    // Safety fallback: if onAuthStateChange never fires (e.g. Supabase SDK
    // error, network completely dead), don't leave the user stuck on the
    // loading spinner forever.
    const fallbackTimer = setTimeout(() => {
      if (mounted && !initialised) {
        console.warn("Auth: onAuthStateChange did not fire within timeout, falling back.");
        initialised = true;
        setLoading(false);
        setRoleLoading(false);
        setIsRestoringSession(false);
      }
    }, AUTH_TIMEOUT_MS);

    return () => {
      mounted = false;
      clearTimeout(fallbackTimer);
      subscription.unsubscribe();
    };
  }, [readCachedUserRole, reconcileSessionRole, setTrackedUser]);

  const signOut = useCallback(async () => {
    setSession(null);
    setTrackedUser(null);
    setTrackedUserRole(null);
    roleRequestIdRef.current += 1;
    setIsRestoringSession(false);

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
      setTrackedUser(null);
      setTrackedUserRole(null);
      clearSupabaseAuthStorage();
    }
  }, [setTrackedUser, setTrackedUserRole]);

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
        roleLoading,
        isRestoringSession,
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
