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

export const clearSupabaseAuthStorage = () => {
  if (typeof window === 'undefined') return;

  const projectRef = getSupabaseProjectRef();
  const storageKeys = [
    SUPABASE_AUTH_STORAGE_KEY,
    projectRef ? `sb-${projectRef}-auth-token` : null,
  ].filter(Boolean) as string[];

  storageKeys.forEach((key) => {
    window.localStorage.removeItem(key);
    window.sessionStorage.removeItem(key);
  });
};

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    storageKey: SUPABASE_AUTH_STORAGE_KEY,
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
    lock: async (_name, _acquireTimeout, fn) => fn(),
  },
});
