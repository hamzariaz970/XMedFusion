export type ThemeColors = {
  background: string;
  backgroundSoft: string;
  backgroundDeep: string;
  backgroundElevated: string;
  foreground: string;
  mutedForeground: string;
  card: string;
  cardForeground: string;
  border: string;
  cardBorder: string;
  input: string;
  primary: string;
  primaryForeground: string;
  primaryGlow: string;
  accent: string;
  accentForeground: string;
  accentGlow: string;
  secondary: string;
  secondaryForeground: string;
  success: string;
  successBg: string;
  warning: string;
  warningBg: string;
  destructive: string;
  destructiveBg: string;
  medicalTeal: string;
  medicalBlue: string;
  medicalBlueLight: string;
  clinicalInk: string;
  clinicalSteel: string;
  clinicalVeil: string;
  white: string;
  black: string;
  transparent: string;
  shadowColor: string;
};

export const lightTheme: ThemeColors = {
  background: 'hsl(0, 0%, 99%)',
  backgroundSoft: 'hsl(224, 56%, 97%)',
  backgroundDeep: 'hsl(224, 56%, 96%)',
  backgroundElevated: 'rgba(255, 255, 255, 0.9)',
  foreground: 'hsl(225, 12%, 16%)',
  mutedForeground: 'hsl(226, 8%, 42%)',
  card: 'hsl(0, 0%, 100%)',
  cardForeground: 'hsl(225, 12%, 16%)',
  border: 'hsl(225, 18%, 88%)',
  cardBorder: 'rgba(219, 225, 240, 0.85)',
  input: 'hsl(225, 18%, 88%)',
  primary: 'hsl(225, 67%, 66%)',
  primaryForeground: 'hsl(0, 0%, 100%)',
  primaryGlow: 'rgba(123, 145, 233, 0.18)',
  accent: 'hsl(206, 77%, 44%)',
  accentForeground: 'hsl(0, 0%, 100%)',
  accentGlow: 'rgba(26, 128, 199, 0.16)',
  secondary: 'hsl(224, 56%, 95%)',
  secondaryForeground: 'hsl(225, 12%, 16%)',
  success: 'hsl(160, 60%, 45%)',
  successBg: 'rgba(32, 174, 127, 0.14)',
  warning: 'hsl(45, 93%, 47%)',
  warningBg: 'rgba(240, 188, 25, 0.16)',
  destructive: 'hsl(0, 84%, 60%)',
  destructiveBg: 'rgba(239, 68, 68, 0.12)',
  medicalTeal: 'hsl(206, 77%, 44%)',
  medicalBlue: 'hsl(225, 67%, 66%)',
  medicalBlueLight: 'hsl(225, 70%, 94%)',
  clinicalInk: 'hsl(225, 12%, 16%)',
  clinicalSteel: 'hsl(214, 18%, 67%)',
  clinicalVeil: 'hsl(203, 22%, 76%)',
  white: '#FFFFFF',
  black: '#000000',
  transparent: 'transparent',
  shadowColor: 'rgba(30, 41, 59, 0.12)',
};

export const darkTheme: ThemeColors = {
  background: 'hsl(210, 40%, 8%)',
  backgroundSoft: 'hsl(210, 38%, 11%)',
  backgroundDeep: 'hsl(210, 42%, 6%)',
  backgroundElevated: 'rgba(18, 28, 40, 0.92)',
  foreground: 'hsl(210, 20%, 98%)',
  mutedForeground: 'hsl(210, 20%, 65%)',
  card: 'hsl(210, 40%, 11%)',
  cardForeground: 'hsl(210, 20%, 98%)',
  border: 'hsl(210, 30%, 20%)',
  cardBorder: 'rgba(49, 66, 87, 0.92)',
  input: 'hsl(210, 30%, 20%)',
  primary: 'hsl(187, 85%, 50%)',
  primaryForeground: 'hsl(210, 40%, 8%)',
  primaryGlow: 'rgba(14, 184, 212, 0.16)',
  accent: 'hsl(220, 70%, 60%)',
  accentForeground: 'hsl(0, 0%, 100%)',
  accentGlow: 'rgba(96, 136, 238, 0.18)',
  secondary: 'hsl(210, 30%, 18%)',
  secondaryForeground: 'hsl(210, 20%, 98%)',
  success: 'hsl(160, 60%, 45%)',
  successBg: 'rgba(34, 180, 120, 0.14)',
  warning: 'hsl(45, 93%, 47%)',
  warningBg: 'rgba(245, 180, 30, 0.16)',
  destructive: 'hsl(0, 62%, 50%)',
  destructiveBg: 'rgba(180, 60, 60, 0.16)',
  medicalTeal: 'hsl(187, 85%, 50%)',
  medicalBlue: 'hsl(220, 70%, 60%)',
  medicalBlueLight: 'rgba(96, 136, 238, 0.18)',
  clinicalInk: 'hsl(210, 20%, 98%)',
  clinicalSteel: 'hsl(210, 20%, 65%)',
  clinicalVeil: 'hsl(210, 30%, 20%)',
  white: '#FFFFFF',
  black: '#000000',
  transparent: 'transparent',
  shadowColor: 'rgba(2, 8, 23, 0.45)',
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const radius = {
  sm: 10,
  md: 16,
  lg: 24,
  xl: 28,
  xxl: 34,
  full: 9999,
};

export const typography = {
  xs: 11,
  sm: 13,
  base: 15,
  lg: 18,
  xl: 22,
  '2xl': 28,
  '3xl': 36,
};

export const fontFamily = {
  regular: 'PlusJakartaSans_400Regular',
  medium: 'PlusJakartaSans_500Medium',
  semiBold: 'PlusJakartaSans_600SemiBold',
  bold: 'PlusJakartaSans_700Bold',
  extraBold: 'PlusJakartaSans_800ExtraBold',
};

export const colors = lightTheme;
