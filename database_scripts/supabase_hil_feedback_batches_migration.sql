-- ============================================================
-- XMedFusion HIL feedback batching + Vision fine-tune queue
-- Run after supabase_schema.sql and supabase_scan_report_integrity_migration.sql.
-- Safe to re-run.
-- ============================================================

CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM public.user_roles
        WHERE user_id = auth.uid()
          AND role = 'admin'
          AND approval_status = 'approved'
    );
$$;

-- Admins need complete access to the clinical feedback pool.
DROP POLICY IF EXISTS "Admins can read feedback" ON public.feedback;
DROP POLICY IF EXISTS "Admins can manage feedback" ON public.feedback;
CREATE POLICY "Admins can manage feedback" ON public.feedback
    FOR ALL
    USING (public.is_admin())
    WITH CHECK (public.is_admin());

-- Admins need enough patient context to show anonymous study metadata.
DROP POLICY IF EXISTS "Admins can manage all patients" ON public.patients;
CREATE POLICY "Admins can manage all patients" ON public.patients
    FOR ALL
    USING (public.is_admin())
    WITH CHECK (public.is_admin());

DROP POLICY IF EXISTS "Admins can read all scans" ON public.medical_scans;
DROP POLICY IF EXISTS "Admins can manage all scans" ON public.medical_scans;
CREATE POLICY "Admins can manage all scans" ON public.medical_scans
    FOR ALL
    USING (public.is_admin())
    WITH CHECK (public.is_admin());

-- Batches group doctor-approved feedback examples before they are queued for model training.
CREATE TABLE IF NOT EXISTS public.hil_feedback_batches (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    admin_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    instructions TEXT DEFAULT '',
    model_target TEXT NOT NULL DEFAULT 'vision_agent',
    status TEXT NOT NULL DEFAULT 'draft',
    queued_job_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.hil_feedback_batches ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Admins can manage hil_feedback_batches" ON public.hil_feedback_batches;
CREATE POLICY "Admins can manage hil_feedback_batches" ON public.hil_feedback_batches
    FOR ALL
    USING (public.is_admin())
    WITH CHECK (public.is_admin());

-- Each feedback example can belong to one active batch at a time.
CREATE TABLE IF NOT EXISTS public.hil_feedback_batch_items (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    batch_id UUID REFERENCES public.hil_feedback_batches(id) ON DELETE CASCADE NOT NULL,
    feedback_id UUID REFERENCES public.feedback(id) ON DELETE CASCADE NOT NULL UNIQUE,
    scan_id UUID REFERENCES public.medical_scans(id) ON DELETE SET NULL,
    doctor_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    item_order INT NOT NULL DEFAULT 0,
    include_original_report BOOLEAN NOT NULL DEFAULT true,
    anonymize_patient BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.hil_feedback_batch_items ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Admins can manage hil_feedback_batch_items" ON public.hil_feedback_batch_items;
CREATE POLICY "Admins can manage hil_feedback_batch_items" ON public.hil_feedback_batch_items
    FOR ALL
    USING (public.is_admin())
    WITH CHECK (public.is_admin());

-- Persistent queue/job audit. The FastAPI worker processes these one at a time.
CREATE TABLE IF NOT EXISTS public.hil_training_jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    batch_id UUID REFERENCES public.hil_feedback_batches(id) ON DELETE SET NULL,
    requested_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    model_target TEXT NOT NULL DEFAULT 'vision_agent',
    status TEXT NOT NULL DEFAULT 'queued',
    sample_count INT NOT NULL DEFAULT 0,
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    result JSONB DEFAULT '{}'::jsonb,
    error TEXT DEFAULT '',
    adapter_output_dir TEXT DEFAULT '',
    backup_dir TEXT DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.hil_training_jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Admins can manage hil_training_jobs" ON public.hil_training_jobs;
CREATE POLICY "Admins can manage hil_training_jobs" ON public.hil_training_jobs
    FOR ALL
    USING (public.is_admin())
    WITH CHECK (public.is_admin());

CREATE INDEX IF NOT EXISTS idx_hil_feedback_batches_status
    ON public.hil_feedback_batches (status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_hil_feedback_batch_items_batch
    ON public.hil_feedback_batch_items (batch_id, item_order);

CREATE INDEX IF NOT EXISTS idx_hil_training_jobs_status
    ON public.hil_training_jobs (status, created_at);

CREATE OR REPLACE FUNCTION public.touch_hil_feedback_batch()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_touch_hil_feedback_batches ON public.hil_feedback_batches;
CREATE TRIGGER trg_touch_hil_feedback_batches
BEFORE UPDATE ON public.hil_feedback_batches
FOR EACH ROW EXECUTE FUNCTION public.touch_hil_feedback_batch();

DROP TRIGGER IF EXISTS trg_touch_hil_feedback_batch_items ON public.hil_feedback_batch_items;
CREATE TRIGGER trg_touch_hil_feedback_batch_items
BEFORE UPDATE ON public.hil_feedback_batch_items
FOR EACH ROW EXECUTE FUNCTION public.touch_hil_feedback_batch();

DROP TRIGGER IF EXISTS trg_touch_hil_training_jobs ON public.hil_training_jobs;
CREATE TRIGGER trg_touch_hil_training_jobs
BEFORE UPDATE ON public.hil_training_jobs
FOR EACH ROW EXECUTE FUNCTION public.touch_hil_feedback_batch();
