-- Allow admins to read all medical scans so the Admin HIL review can show previews.
DROP POLICY IF EXISTS "Admins can read all scans" ON public.medical_scans;
CREATE POLICY "Admins can read all scans" ON public.medical_scans
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1
            FROM public.user_roles ur
            WHERE ur.user_id = auth.uid()
              AND ur.role = 'admin'
              AND ur.approval_status = 'approved'
        )
    );

-- Backfill stale doctor profiles where the auth user changed but the old email row remained.
UPDATE public.doctors d
SET user_id = au.id,
    status = CASE
        WHEN d.status IS NULL OR d.status = 'pre-approved' THEN 'active'
        ELSE d.status
    END
FROM auth.users au
JOIN public.user_roles ur
  ON ur.user_id = au.id
 AND ur.role = 'doctor'
 AND ur.approval_status = 'approved'
WHERE d.email = au.email
  AND d.user_id IS DISTINCT FROM au.id
  AND NOT EXISTS (
      SELECT 1
      FROM public.doctors dx
      WHERE dx.user_id = au.id
  );

-- Insert a doctor profile for any approved doctor auth user still missing one entirely.
INSERT INTO public.doctors (user_id, full_name, email, specialization, status)
SELECT
    au.id,
    COALESCE(
        NULLIF(TRIM(COALESCE(au.raw_user_meta_data ->> 'full_name', '')), ''),
        SPLIT_PART(au.email, '@', 1)
    ) AS full_name,
    au.email,
    'Radiology' AS specialization,
    'active' AS status
FROM auth.users au
JOIN public.user_roles ur
  ON ur.user_id = au.id
 AND ur.role = 'doctor'
 AND ur.approval_status = 'approved'
LEFT JOIN public.doctors d_user
  ON d_user.user_id = au.id
LEFT JOIN public.doctors d_email
  ON d_email.email = au.email
WHERE d_user.id IS NULL
  AND d_email.id IS NULL
  AND au.email IS NOT NULL;
