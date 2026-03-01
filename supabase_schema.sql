-- Create a patients table
CREATE TABLE public.patients (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) NOT NULL, -- The doctor/user that owns this patient
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    conditions TEXT[] DEFAULT '{}',
    notes TEXT,
    status TEXT DEFAULT 'active', -- active, follow-up, resolved, critical
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Enable RLS for patients
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;

-- Allow users to operate on their own patients
CREATE POLICY "Users can manage their own patients" ON public.patients
    FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Create a medical_scans table
CREATE TABLE public.medical_scans (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    patient_id UUID REFERENCES public.patients(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    scan_type TEXT NOT NULL,
    original_image_url TEXT,
    heatmap_image_url TEXT,
    findings TEXT,
    impression TEXT,
    recommendation TEXT,
    labels TEXT[] DEFAULT '{}',
    kg_data JSONB,
    severity TEXT DEFAULT 'moderate',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Enable RLS for medical_scans
ALTER TABLE public.medical_scans ENABLE ROW LEVEL SECURITY;

-- Allow users to operate on their own scans
CREATE POLICY "Users can manage their own scans" ON public.medical_scans
    FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Create Storage Bucket for medical images
INSERT INTO storage.buckets (id, name, public) 
VALUES ('medical-images', 'medical-images', true)
ON CONFLICT (id) DO NOTHING;

-- Storage Policies for medical-images
-- (In a real hospital app this should be private, but for ease here we make it public reads, authenticated uploads)
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
