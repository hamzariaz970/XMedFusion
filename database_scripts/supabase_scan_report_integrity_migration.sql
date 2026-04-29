-- ============================================================
-- XMedFusion scan/report integrity migration
-- Run after supabase_schema.sql. Safe to re-run.
-- ============================================================

-- Keep a proper update timestamp on scan rows.
ALTER TABLE public.medical_scans
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE
DEFAULT timezone('utc'::text, now()) NOT NULL;

CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_medical_scans_updated_at ON public.medical_scans;
CREATE TRIGGER trg_medical_scans_updated_at
BEFORE UPDATE ON public.medical_scans
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Ensure scan rows cannot be attached to another doctor's patient.
DROP POLICY IF EXISTS "Users can manage their own scans" ON public.medical_scans;
DROP POLICY IF EXISTS "Doctors can manage scans for own patients" ON public.medical_scans;

CREATE POLICY "Doctors can manage scans for own patients" ON public.medical_scans
    FOR ALL
    USING (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1
            FROM public.patients p
            WHERE p.id = medical_scans.patient_id
              AND p.user_id = auth.uid()
        )
    )
    WITH CHECK (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1
            FROM public.patients p
            WHERE p.id = medical_scans.patient_id
              AND p.user_id = auth.uid()
        )
    );

-- Touch the patient record whenever its scan/report changes so "last visit"
-- and dashboard recency are based on diagnostic activity.
CREATE OR REPLACE FUNCTION public.touch_patient_from_scan()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
AS $$
BEGIN
    UPDATE public.patients
    SET updated_at = timezone('utc'::text, now())
    WHERE id = NEW.patient_id;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_touch_patient_from_scan ON public.medical_scans;
CREATE TRIGGER trg_touch_patient_from_scan
AFTER INSERT OR UPDATE ON public.medical_scans
FOR EACH ROW EXECUTE FUNCTION public.touch_patient_from_scan();

-- Optional audit table for doctor review/edit actions. The app updates the
-- medical_scans row first, then writes here when this table exists.
CREATE TABLE IF NOT EXISTS public.feedback (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    scan_id UUID REFERENCES public.medical_scans(id) ON DELETE SET NULL,
    doctor_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    original_findings TEXT DEFAULT '',
    edited_findings TEXT DEFAULT '',
    original_impression TEXT DEFAULT '',
    edited_impression TEXT DEFAULT '',
    edited_recommendation TEXT DEFAULT '',
    edited_labels TEXT[] DEFAULT '{}',
    edited_kg JSONB,
    doctor_notes TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.feedback ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.user_roles
        WHERE user_id = auth.uid()
          AND role = 'admin'
          AND approval_status = 'approved'
    );
$$;

DROP POLICY IF EXISTS "Doctors can manage own feedback" ON public.feedback;
DROP POLICY IF EXISTS "Admins can read feedback" ON public.feedback;

CREATE POLICY "Doctors can manage own feedback" ON public.feedback
    FOR ALL
    USING (doctor_id = auth.uid())
    WITH CHECK (
        doctor_id = auth.uid()
        AND (
            scan_id IS NULL
            OR EXISTS (
                SELECT 1
                FROM public.medical_scans ms
                WHERE ms.id = scan_id
                  AND ms.user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Admins can read feedback" ON public.feedback
    FOR SELECT USING (public.is_admin());

CREATE INDEX IF NOT EXISTS idx_medical_scans_patient_created
    ON public.medical_scans (patient_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_medical_scans_user_created
    ON public.medical_scans (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_feedback_scan_id
    ON public.feedback (scan_id);
