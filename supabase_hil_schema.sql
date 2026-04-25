-- ============================================================
-- HIL (Human-in-the-Loop) Schema Migration  (v2 — Hardened)
-- Run this in Supabase SQL Editor AFTER the main schema
-- ============================================================
-- Changes from v1:
--   • Role-scoped RLS policies (no more blanket 'authenticated')
--   • added updated_at to hil_scans
--   • DB trigger auto-updates hil_tasks.completed_scans + status

-- ────────────────────────────────────────────────────
-- Helper: check if the calling user has admin role
-- ────────────────────────────────────────────────────
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

-- ============================================================
-- 1. HIL Tasks — Admin-created labeling assignments
-- ============================================================
CREATE TABLE IF NOT EXISTS public.hil_tasks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    admin_id UUID REFERENCES auth.users(id) NOT NULL,
    doctor_id UUID REFERENCES auth.users(id) NOT NULL,
    title TEXT NOT NULL,
    instructions TEXT DEFAULT '',
    total_scans INT NOT NULL DEFAULT 0,
    completed_scans INT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'assigned',  -- assigned | in_progress | completed | reviewed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.hil_tasks ENABLE ROW LEVEL SECURITY;

-- Drop old blanket policies if they exist
DROP POLICY IF EXISTS "Authenticated users can read hil_tasks" ON public.hil_tasks;
DROP POLICY IF EXISTS "Authenticated users can insert hil_tasks" ON public.hil_tasks;
DROP POLICY IF EXISTS "Authenticated users can update hil_tasks" ON public.hil_tasks;
DROP POLICY IF EXISTS "Authenticated users can delete hil_tasks" ON public.hil_tasks;

-- Admins: full CRUD
CREATE POLICY "Admins can manage hil_tasks" ON public.hil_tasks
    FOR ALL USING (public.is_admin());

-- Doctors: can only read tasks assigned to them
CREATE POLICY "Doctors can read own tasks" ON public.hil_tasks
    FOR SELECT USING (doctor_id = auth.uid());


-- ============================================================
-- 2. HIL Scans — Individual scans within a task
-- ============================================================
CREATE TABLE IF NOT EXISTS public.hil_scans (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    task_id UUID REFERENCES public.hil_tasks(id) ON DELETE CASCADE NOT NULL,
    image_url TEXT NOT NULL,
    scan_order INT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | labeled | approved | rejected
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.hil_scans ENABLE ROW LEVEL SECURITY;

-- Drop old blanket policies if they exist
DROP POLICY IF EXISTS "Authenticated users can read hil_scans" ON public.hil_scans;
DROP POLICY IF EXISTS "Authenticated users can insert hil_scans" ON public.hil_scans;
DROP POLICY IF EXISTS "Authenticated users can update hil_scans" ON public.hil_scans;
DROP POLICY IF EXISTS "Authenticated users can delete hil_scans" ON public.hil_scans;

-- Admins: full CRUD
CREATE POLICY "Admins can manage hil_scans" ON public.hil_scans
    FOR ALL USING (public.is_admin());

-- Doctors: can read scans belonging to their assigned tasks
CREATE POLICY "Doctors can read scans for own tasks" ON public.hil_scans
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.hil_tasks
            WHERE id = hil_scans.task_id
              AND doctor_id = auth.uid()
        )
    );

-- Doctors: can update scan status (e.g., pending → labeled) for own tasks
CREATE POLICY "Doctors can update scans for own tasks" ON public.hil_scans
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM public.hil_tasks
            WHERE id = hil_scans.task_id
              AND doctor_id = auth.uid()
        )
    );


-- ============================================================
-- 3. HIL Reports — Doctor-submitted reports per scan
-- ============================================================
CREATE TABLE IF NOT EXISTS public.hil_reports (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    scan_id UUID REFERENCES public.hil_scans(id) ON DELETE CASCADE NOT NULL,
    task_id UUID REFERENCES public.hil_tasks(id) ON DELETE CASCADE NOT NULL,
    doctor_id UUID REFERENCES auth.users(id) NOT NULL,
    indication TEXT DEFAULT '',
    comparison TEXT DEFAULT '',
    findings TEXT NOT NULL DEFAULT '',
    impression TEXT NOT NULL DEFAULT '',
    admin_feedback TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'submitted',  -- draft | submitted | approved | rejected
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

ALTER TABLE public.hil_reports ENABLE ROW LEVEL SECURITY;

-- Drop old blanket policies if they exist
DROP POLICY IF EXISTS "Authenticated users can read hil_reports" ON public.hil_reports;
DROP POLICY IF EXISTS "Authenticated users can insert hil_reports" ON public.hil_reports;
DROP POLICY IF EXISTS "Authenticated users can update hil_reports" ON public.hil_reports;
DROP POLICY IF EXISTS "Authenticated users can delete hil_reports" ON public.hil_reports;

-- Admins: full CRUD (needed for approve/reject + feedback)
CREATE POLICY "Admins can manage hil_reports" ON public.hil_reports
    FOR ALL USING (public.is_admin());

-- Doctors: can read their own reports
CREATE POLICY "Doctors can read own reports" ON public.hil_reports
    FOR SELECT USING (doctor_id = auth.uid());

-- Doctors: can insert reports for themselves only
CREATE POLICY "Doctors can insert own reports" ON public.hil_reports
    FOR INSERT WITH CHECK (doctor_id = auth.uid());

-- Doctors: can update their own reports (drafts, rejected → re-submit)
CREATE POLICY "Doctors can update own reports" ON public.hil_reports
    FOR UPDATE USING (doctor_id = auth.uid());


-- ============================================================
-- 4. Trigger: Auto-update hil_tasks progress on report changes
-- ============================================================
CREATE OR REPLACE FUNCTION public.update_hil_task_progress()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
AS $$
BEGIN
    UPDATE public.hil_tasks SET
        completed_scans = (
            SELECT COUNT(DISTINCT scan_id)
            FROM public.hil_reports
            WHERE task_id = NEW.task_id
              AND status IN ('submitted', 'approved')
        ),
        status = CASE
            WHEN (
                SELECT COUNT(DISTINCT scan_id)
                FROM public.hil_reports
                WHERE task_id = NEW.task_id
                  AND status IN ('submitted', 'approved')
            ) >= total_scans THEN 'completed'
            WHEN (
                SELECT COUNT(DISTINCT scan_id)
                FROM public.hil_reports
                WHERE task_id = NEW.task_id
                  AND status IN ('submitted', 'approved')
            ) > 0 THEN 'in_progress'
            ELSE 'assigned'
        END,
        updated_at = now()
    WHERE id = NEW.task_id;
    RETURN NEW;
END;
$$;

-- Drop if exists to make this script idempotent
DROP TRIGGER IF EXISTS trg_hil_report_progress ON public.hil_reports;

CREATE TRIGGER trg_hil_report_progress
AFTER INSERT OR UPDATE ON public.hil_reports
FOR EACH ROW EXECUTE FUNCTION public.update_hil_task_progress();
