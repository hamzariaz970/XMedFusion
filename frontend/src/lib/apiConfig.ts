/**
 * API Configuration Utility
 * Handles fallback between Ngrok and Localhost
 */

const NGROK_URL = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_BASE_URL_NGROK;
const LOCAL_URL = import.meta.env.VITE_API_BASE_URL_LOCAL || "http://localhost:8000";
const LOOPBACK_URL = "http://127.0.0.1:8000";

let cachedBaseUrl: string | null = null;

const getCandidateBaseUrls = () => {
  const candidates = [cachedBaseUrl, NGROK_URL, LOCAL_URL, LOOPBACK_URL]
    .filter(Boolean)
    .map((url) => (url as string).replace(/\/$/, ""));

  return Array.from(new Set(candidates));
};

const canReachApi = async (baseUrl: string, timeoutMs = 2500) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const response = await fetch(`${baseUrl}/api/health`, {
      method: "GET",
      signal: controller.signal,
      headers: {
        "ngrok-skip-browser-warning": "true",
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
      if (candidateUrl === NGROK_URL) {
        console.log("🚀 Using Ngrok API:", candidateUrl);
      } else {
        console.log("🏠 Using Local API:", candidateUrl);
      }
      cachedBaseUrl = candidateUrl;
      return candidateUrl;
    }
  }

  cachedBaseUrl = LOCAL_URL;
  console.log("🏠 Using Local API:", LOCAL_URL);
  return LOCAL_URL;
};

/**
 * Helper to get the base URL synchronously if already cached, 
 * otherwise returns local as a safe default while the check happens.
 */
export const getApiBaseSync = () => cachedBaseUrl || LOCAL_URL;
