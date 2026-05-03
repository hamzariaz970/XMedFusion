-- ============================================================
-- XMedFusion CT multi-view support
-- Adds structured storage for multi-slice scans and explainability references.
-- Safe to re-run.
-- ============================================================

ALTER TABLE public.medical_scans
ADD COLUMN IF NOT EXISTS source_images JSONB DEFAULT '[]'::jsonb;

ALTER TABLE public.medical_scans
ADD COLUMN IF NOT EXISTS explainability_reference_image_url TEXT;

ALTER TABLE public.medical_scans
ADD COLUMN IF NOT EXISTS scan_metadata JSONB DEFAULT '{}'::jsonb;
