-- ============================================================
-- Bridge approved legacy HIL task reports into Vision HIL batches.
-- Safe to re-run.
-- ============================================================

ALTER TABLE public.hil_feedback_batch_items
ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'feedback';

ALTER TABLE public.hil_feedback_batch_items
ADD COLUMN IF NOT EXISTS hil_report_id UUID REFERENCES public.hil_reports(id) ON DELETE CASCADE;

ALTER TABLE public.hil_feedback_batch_items
ADD COLUMN IF NOT EXISTS hil_scan_id UUID REFERENCES public.hil_scans(id) ON DELETE SET NULL;

ALTER TABLE public.hil_feedback_batch_items
ALTER COLUMN feedback_id DROP NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_hil_feedback_batch_items_hil_report_unique
    ON public.hil_feedback_batch_items (hil_report_id)
    WHERE hil_report_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_hil_feedback_batch_items_hil_scan
    ON public.hil_feedback_batch_items (hil_scan_id)
    WHERE hil_scan_id IS NOT NULL;
