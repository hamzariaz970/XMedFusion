-- ============================================================
-- MIGRATION: Run this in Supabase SQL Editor
-- (You already have patients, medical_scans, and storage set up)
-- ============================================================

-- 1. Drop the old doctors table & its policies (recreate with proper RLS)
DROP POLICY IF EXISTS "Allow all operations on doctors" ON public.doctors;
DROP TABLE IF EXISTS public.doctors;

-- 2. Create user_roles table
CREATE TABLE public.user_roles (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    role TEXT NOT NULL DEFAULT 'doctor',
    approval_status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own role" ON public.user_roles
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Admins can read all roles" ON public.user_roles
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM public.user_roles ur WHERE ur.user_id = auth.uid() AND ur.role = 'admin')
    );

CREATE POLICY "Admins can update roles" ON public.user_roles
    FOR UPDATE USING (
        EXISTS (SELECT 1 FROM public.user_roles ur WHERE ur.user_id = auth.uid() AND ur.role = 'admin')
    );

CREATE POLICY "Users can insert own role" ON public.user_roles
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Admins can insert roles" ON public.user_roles
    FOR INSERT WITH CHECK (
        EXISTS (SELECT 1 FROM public.user_roles ur WHERE ur.user_id = auth.uid() AND ur.role = 'admin')
    );

CREATE POLICY "Admins can delete roles" ON public.user_roles
    FOR DELETE USING (
        EXISTS (SELECT 1 FROM public.user_roles ur WHERE ur.user_id = auth.uid() AND ur.role = 'admin')
    );

-- 3. Recreate doctors table with proper RLS
CREATE TABLE public.doctors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    specialization TEXT NOT NULL DEFAULT 'Radiology',
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.doctors ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own doctor profile" ON public.doctors
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Admins can manage doctors" ON public.doctors
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.user_roles ur WHERE ur.user_id = auth.uid() AND ur.role = 'admin')
    );

CREATE POLICY "Users can insert own doctor profile" ON public.doctors
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- 4. Seed your existing admin user
-- IMPORTANT: Replace 'YOUR_AUTH_USER_ID' with your actual auth.users id
-- You can find it in Supabase Dashboard > Authentication > Users
-- INSERT INTO public.user_roles (user_id, role, approval_status)
-- VALUES ('YOUR_AUTH_USER_ID', 'admin', 'approved');
