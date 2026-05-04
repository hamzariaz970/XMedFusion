import Constants from 'expo-constants';

const NGROK_URL = process.env.EXPO_PUBLIC_API_BASE_URL_NGROK;
const DEVICE_URL = process.env.EXPO_PUBLIC_API_BASE_URL_DEVICE;
const LOCAL_URL = process.env.EXPO_PUBLIC_API_BASE_URL_LOCAL || 'http://127.0.0.1:8000';
const LOOPBACK_URL = 'http://127.0.0.1:8000';
const ANDROID_EMULATOR_URL = 'http://10.0.2.2:8000';

let cachedBaseUrl: string | null = null;

const normalizeUrl = (url: string) => url.replace(/\/$/, '');

const inferExpoHostUrl = () => {
  const hostUri =
    Constants.expoConfig?.hostUri ||
    (Constants as any).manifest2?.extra?.expoGo?.debuggerHost ||
    (Constants as any).manifest?.debuggerHost ||
    '';

  if (!hostUri) return null;

  const host = String(hostUri).split(':')[0]?.trim();
  if (!host) return null;

  return `http://${host}:8000`;
};

const getCandidateBaseUrls = () => {
  const candidates = [cachedBaseUrl, DEVICE_URL, inferExpoHostUrl(), NGROK_URL, LOCAL_URL, LOOPBACK_URL, ANDROID_EMULATOR_URL]
    .filter(Boolean)
    .map((url) => normalizeUrl(url as string));

  return Array.from(new Set(candidates));
};

const canReachApi = async (baseUrl: string, timeoutMs = 2500) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const response = await fetch(`${baseUrl}/api/health`, {
      method: 'GET',
      signal: controller.signal,
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    });

    clearTimeout(timeoutId);
    return response.ok;
  } catch {
    return false;
  }
};

export const getApiBase = async (forceRefresh = false): Promise<string> => {
  if (cachedBaseUrl && !forceRefresh && await canReachApi(cachedBaseUrl, 1200)) {
    return cachedBaseUrl;
  }

  for (const candidateUrl of getCandidateBaseUrls()) {
    if (await canReachApi(candidateUrl)) {
      cachedBaseUrl = candidateUrl;
      return candidateUrl;
    }
  }

  cachedBaseUrl = normalizeUrl(LOCAL_URL);
  return cachedBaseUrl;
};

export const getApiBaseSync = () => cachedBaseUrl || normalizeUrl(LOCAL_URL);
