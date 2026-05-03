import { createClient } from '@supabase/supabase-js';

// Replace these with your Supabase Project URL and Anon Key
// In a production app, use import.meta.env.VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://ozasblnmcujqiirdjcxj.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'sb_publishable_oSPQKYq5SC89YU0p0XnuaQ_4rc4sFB6';
export const SUPABASE_AUTH_STORAGE_KEY = 'xmedfusion-auth';

const getSupabaseProjectRef = () => {
  try {
    return new URL(supabaseUrl).hostname.split('.')[0];
  } catch {
    return null;
  }
};

export const hasPersistedSupabaseSession = () => {
  if (typeof window === 'undefined') return false;

  const projectRef = getSupabaseProjectRef();
  const candidateKeys = [
    SUPABASE_AUTH_STORAGE_KEY,
    projectRef ? `sb-${projectRef}-auth-token` : null,
  ].filter(Boolean) as string[];

  return candidateKeys.some((key) =>
    window.localStorage.getItem(key) || window.sessionStorage.getItem(key)
  );
};

export const clearSupabaseAuthStorage = () => {
  if (typeof window === 'undefined') return;

  const projectRef = getSupabaseProjectRef();

  // Explicit known keys
  const explicitKeys = [
    SUPABASE_AUTH_STORAGE_KEY,
    projectRef ? `sb-${projectRef}-auth-token` : null,
  ].filter(Boolean) as string[];

  // Also sweep all `sb-*` prefixed keys (PKCE verifiers, refresh token cache, etc.)
  // that the Supabase SDK may have written during the session.
  const sweepStorage = (storage: Storage) => {
    const keysToRemove: string[] = [];
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key && (explicitKeys.includes(key) || key.startsWith('sb-'))) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach((key) => storage.removeItem(key));
  };

  sweepStorage(window.localStorage);
  sweepStorage(window.sessionStorage);
};

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    storageKey: SUPABASE_AUTH_STORAGE_KEY,
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
  },
});
