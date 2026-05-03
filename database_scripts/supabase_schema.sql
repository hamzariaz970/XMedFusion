-- ============================================================
-- XMedFusion Supabase Schema
-- ============================================================

-- ── 1. User Roles ─────────────────────────────────────────────
-- Tracks every authenticated user's role and approval status.
CREATE TABLE public.user_roles (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    role TEXT NOT NULL DEFAULT 'doctor',           -- 'admin' | 'doctor'
    approval_status TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'approved' | 'rejected'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

-- Everyone authenticated can read their own role
CREATE POLICY "Users can read own role" ON public.user_roles
    FOR SELECT USING (auth.uid() = user_id);

-- Admins can read all roles
CREATE POLICY "Admins can read all roles" ON public.user_roles
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.user_roles ur
            WHERE ur.user_id = auth.uid() AND ur.role = 'admin'
        )
    );

-- Admins can update any role (approve/reject/promote)
CREATE POLICY "Admins can update roles" ON public.user_roles
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM public.user_roles ur
            WHERE ur.user_id = auth.uid() AND ur.role = 'admin'
        )
    );

-- Anyone authenticated can insert their own role row (on signup)
CREATE POLICY "Users can insert own role" ON public.user_roles
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Admins can insert roles for others (adding new admins)
CREATE POLICY "Admins can insert roles" ON public.user_roles
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.user_roles ur
            WHERE ur.user_id = auth.uid() AND ur.role = 'admin'
        )
    );

-- Admins can delete roles
CREATE POLICY "Admins can delete roles" ON public.user_roles
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM public.user_roles ur
            WHERE ur.user_id = auth.uid() AND ur.role = 'admin'
        )
    );


-- ── 2. Doctors ────────────────────────────────────────────────
CREATE TABLE public.doctors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    specialization TEXT NOT NULL DEFAULT 'Radiology',
    status TEXT DEFAULT 'active',                    -- 'active' | 'suspended'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.doctors ENABLE ROW LEVEL SECURITY;

-- Users can read their own doctor profile
CREATE POLICY "Users can read own doctor profile" ON public.doctors
    FOR SELECT USING (auth.uid() = user_id);

-- Admins can do everything on doctors
CREATE POLICY "Admins can manage doctors" ON public.doctors
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.user_roles ur
            WHERE ur.user_id = auth.uid() AND ur.role = 'admin'
        )
    );

-- Users can insert their own doctor profile (signup)
CREATE POLICY "Users can insert own doctor profile" ON public.doctors
    FOR INSERT WITH CHECK (auth.uid() = user_id);


-- ── 3. Patients ───────────────────────────────────────────────
CREATE TABLE public.patients (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    conditions TEXT[] DEFAULT '{}',
    notes TEXT,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage their own patients" ON public.patients
    FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ── 4. Medical Scans ──────────────────────────────────────────
CREATE TABLE public.medical_scans (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    patient_id UUID REFERENCES public.patients(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    scan_type TEXT NOT NULL,
    original_image_url TEXT,
    source_images JSONB DEFAULT '[]'::jsonb,
    heatmap_image_url TEXT,
    explainability_reference_image_url TEXT,
    findings TEXT,
    impression TEXT,
    recommendation TEXT,
    labels TEXT[] DEFAULT '{}',
    kg_data JSONB,
    scan_metadata JSONB DEFAULT '{}'::jsonb,
    severity TEXT DEFAULT 'moderate',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.medical_scans ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage their own scans" ON public.medical_scans
    FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ── 5. Storage ────────────────────────────────────────────────
INSERT INTO storage.buckets (id, name, public)
VALUES ('medical-images', 'medical-images', true)
ON CONFLICT (id) DO NOTHING;

CREATE POLICY "Anyone can view medical-images" ON storage.objects
    FOR SELECT
    USING (bucket_id = 'medical-images');

CREATE POLICY "Authenticated users can upload medical-images" ON storage.objects
    FOR INSERT
    WITH CHECK (bucket_id = 'medical-images' AND auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can update own medical-images" ON storage.objects
    FOR UPDATE
    USING (bucket_id = 'medical-images' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Authenticated users can delete own medical-images" ON storage.objects
    FOR DELETE
    USING (bucket_id = 'medical-images' AND auth.uid()::text = (storage.foldername(name))[1]);
