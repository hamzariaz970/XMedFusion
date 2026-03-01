import { useEffect, useState } from "react";
import { Navigate, Outlet } from "react-router-dom";
import { supabase } from "@/lib/supabaseClient";

export const ProtectedRoute = () => {
    const [session, setSession] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        supabase.auth.getSession().then(({ data: { session } }) => {
            setSession(session);
            setLoading(false);
        });

        const {
            data: { subscription },
        } = supabase.auth.onAuthStateChange((_event, session) => {
            setSession(session);
        });

        return () => subscription.unsubscribe();
    }, []);

    if (loading) {
        return <div className="flex h-screen items-center justify-center">Loading...</div>;
    }

    if (!session) {
        return <Navigate to="/" replace />;
    }

    return <Outlet />;
};
