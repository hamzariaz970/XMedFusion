import 'react-native-url-polyfill/auto';
import * as SecureStore from 'expo-secure-store';
import { createClient } from '@supabase/supabase-js';

// Keep mobile pointed at the same Supabase project as the web app.
const supabaseUrl = 'https://ozasblnmcujqiirdjcxj.supabase.co';
const supabaseAnonKey = 'sb_publishable_oSPQKYq5SC89YU0p0XnuaQ_4rc4sFB6';

// Robust storage adapter for Expo
const ExpoSecureStoreAdapter = {
    getItem: (key: string) => {
        return SecureStore.getItemAsync(key);
    },
    setItem: (key: string, value: string) => {
        return SecureStore.setItemAsync(key, value);
    },
    removeItem: (key: string) => {
        return SecureStore.deleteItemAsync(key);
    },
};

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
        storage: ExpoSecureStoreAdapter,
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: false,
    },
});
