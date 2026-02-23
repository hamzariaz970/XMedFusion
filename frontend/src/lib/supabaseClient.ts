import { createClient } from '@supabase/supabase-js';

// Replace these with your Supabase Project URL and Anon Key
// In a production app, use import.meta.env.VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://hjyjzcjhgbqssarkwmki.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhqeWp6Y2poZ2Jxc3Nhcmt3bWtpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE4NTM4NzYsImV4cCI6MjA4NzQyOTg3Nn0.7KY-perrES8-fnlaow6lhJ4SRADXgIFTf0qDt3hFpiA';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
