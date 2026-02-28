// XMedFusion — Dual Theme Color System
// Light & dark values sourced directly from frontend/src/index.css
// :root = light, .dark = dark

export type ThemeColors = {
    background: string;
    backgroundDeep: string;
    backgroundCard: string;
    card: string;
    cardForeground: string;
    foreground: string;
    mutedForeground: string;
    primary: string;
    primaryForeground: string;
    primaryGlow: string;
    accent: string;
    accentForeground: string;
    border: string;
    cardBorder: string;
    input: string;
    destructive: string;
    destructiveBg: string;
    success: string;
    successBg: string;
    warning: string;
    warningBg: string;
    transparent: string;
    white: string;
    black: string;
};

// ─── DARK THEME (from .dark in index.css) ─────────────────────────────
export const darkTheme: ThemeColors = {
    background: 'hsl(210, 40%, 8%)',
    card: 'hsl(210, 40%, 11%)',
    backgroundDeep: 'hsl(210, 40%, 6%)',
    backgroundCard: 'hsl(210, 30%, 18%)',
    foreground: 'hsl(210, 20%, 98%)',
    cardForeground: 'hsl(210, 20%, 98%)',
    mutedForeground: 'hsl(210, 20%, 65%)',
    primary: 'hsl(187, 85%, 50%)',
    primaryForeground: 'hsl(210, 40%, 8%)',
    primaryGlow: 'rgba(14, 184, 212, 0.12)',
    accent: 'hsl(220, 70%, 60%)',
    accentForeground: 'hsl(0, 0%, 100%)',
    border: 'hsl(210, 30%, 20%)',
    cardBorder: 'hsl(210, 30%, 20%)',
    input: 'hsl(210, 30%, 20%)',
    destructive: 'hsl(0, 62%, 50%)',
    destructiveBg: 'rgba(180, 60, 60, 0.15)',
    success: 'hsl(160, 60%, 45%)',
    successBg: 'rgba(34, 180, 120, 0.12)',
    warning: 'hsl(45, 93%, 47%)',
    warningBg: 'rgba(245, 180, 30, 0.12)',
    transparent: 'transparent',
    white: '#FFFFFF',
    black: '#000000',
};

// ─── LIGHT THEME (from :root in index.css) ─────────────────────────────
export const lightTheme: ThemeColors = {
    background: 'hsl(210, 20%, 98%)',
    card: 'hsl(0, 0%, 100%)',
    backgroundDeep: 'hsl(210, 20%, 95%)',
    backgroundCard: 'hsl(210, 30%, 95%)',
    foreground: 'hsl(210, 40%, 11%)',
    cardForeground: 'hsl(210, 40%, 11%)',
    mutedForeground: 'hsl(210, 20%, 45%)',
    primary: 'hsl(187, 85%, 43%)',
    primaryForeground: 'hsl(0, 0%, 100%)',
    primaryGlow: 'rgba(14, 184, 180, 0.10)',
    accent: 'hsl(220, 70%, 55%)',
    accentForeground: 'hsl(0, 0%, 100%)',
    border: 'hsl(210, 20%, 90%)',
    cardBorder: 'hsl(210, 20%, 88%)',
    input: 'hsl(210, 20%, 90%)',
    destructive: 'hsl(0, 84%, 60%)',
    destructiveBg: 'rgba(220, 50, 50, 0.08)',
    success: 'hsl(160, 60%, 40%)',
    successBg: 'rgba(20, 160, 100, 0.10)',
    warning: 'hsl(45, 93%, 40%)',
    warningBg: 'rgba(200, 150, 10, 0.10)',
    transparent: 'transparent',
    white: '#FFFFFF',
    black: '#000000',
};

// ─── SPACING ────────────────────────────────────────────────────────────
export const spacing = {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
};

// ─── BORDER RADIUS ──────────────────────────────────────────────────────
export const radius = {
    sm: 8,
    md: 10,
    lg: 12,
    xl: 20,
    xxl: 28,
    full: 9999,
};

// ─── TYPOGRAPHY ─────────────────────────────────────────────────────────
export const typography = {
    xs: 11,
    sm: 13,
    base: 15,
    lg: 17,
    xl: 20,
    '2xl': 24,
    '3xl': 30,
};

// ─── FONT FAMILIES ──────────────────────────────────────────────────────
export const fontFamily = {
    regular: 'PlusJakartaSans_400Regular',
    medium: 'PlusJakartaSans_500Medium',
    semiBold: 'PlusJakartaSans_600SemiBold',
    bold: 'PlusJakartaSans_700Bold',
    extraBold: 'PlusJakartaSans_800ExtraBold',
};

// Legacy default export for backward compat (dark)
export const colors = darkTheme;
