import type { ThemeColors } from './colors';
import { radius, spacing } from './colors';

export const createShadow = (theme: ThemeColors, level: 'sm' | 'md' | 'lg' = 'md') => {
  const shadow =
    level === 'sm'
      ? '0 8px 20px rgba(15, 23, 42, 0.08)'
      : level === 'lg'
        ? '0 26px 80px rgba(15, 23, 42, 0.18)'
        : '0 12px 32px rgba(15, 23, 42, 0.12)';

  return {
    boxShadow: shadow,
  } as const;
};

export const shellBackground = (theme: ThemeColors) => ({
  backgroundColor: theme.background,
});

export const glassPanel = (theme: ThemeColors) => ({
  backgroundColor: theme.backgroundElevated,
  borderColor: 'rgba(255,255,255,0.7)',
  borderWidth: 1,
  borderRadius: radius.xl,
  ...createShadow(theme, 'md'),
});

export const surfaceCard = (theme: ThemeColors) => ({
  backgroundColor: theme.backgroundElevated,
  borderColor: theme.cardBorder,
  borderWidth: 1,
  borderRadius: radius.lg,
  ...createShadow(theme, 'sm'),
});

export const mutedCard = (theme: ThemeColors) => ({
  backgroundColor: theme.secondary,
  borderColor: theme.cardBorder,
  borderWidth: 1,
  borderRadius: radius.md,
});

export const sectionPadding = {
  paddingHorizontal: spacing.lg,
  paddingBottom: spacing.xxl,
};

export const statusPill = (backgroundColor: string) => ({
  backgroundColor,
  borderRadius: radius.full,
  paddingHorizontal: spacing.sm,
  paddingVertical: 4,
});
