import { createClient } from '@supabase/supabase-js';

// Replace these with your Supabase Project URL and Anon Key
// In a production app, use import.meta.env.VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://ozasblnmcujqiirdjcxj.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'sb_publishable_oSPQKYq5SC89YU0p0XnuaQ_4rc4sFB6';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
