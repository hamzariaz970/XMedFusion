/**
 * API Configuration Utility
 * Handles fallback between Ngrok and Localhost
 */

const NGROK_URL = import.meta.env.VITE_API_BASE_URL_NGROK;
const LOCAL_URL = import.meta.env.VITE_API_BASE_URL_LOCAL || "http://localhost:8000";

let cachedBaseUrl: string | null = null;

export const getApiBase = async (forceRefresh = false): Promise<string> => {
  if (cachedBaseUrl && !forceRefresh) {
    return cachedBaseUrl;
  }

  if (!NGROK_URL) {
    cachedBaseUrl = LOCAL_URL;
    return LOCAL_URL;
  }

  try {
    // Try to ping the health endpoint of the ngrok URL
    // We use a short timeout to avoid hanging the UI
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    const response = await fetch(`${NGROK_URL}/api/health`, {
      method: 'GET',
      signal: controller.signal,
      headers: {
        "ngrok-skip-browser-warning": "true"
      }
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      console.log("🚀 Using Ngrok API:", NGROK_URL);
      cachedBaseUrl = NGROK_URL;
      return NGROK_URL;
    }
  } catch (error) {
    console.warn("⚠️ Ngrok connection failed, falling back to localhost:", error);
  }

  console.log("🏠 Using Local API:", LOCAL_URL);
  cachedBaseUrl = LOCAL_URL;
  return LOCAL_URL;
};

/**
 * Helper to get the base URL synchronously if already cached, 
 * otherwise returns local as a safe default while the check happens.
 */
export const getApiBaseSync = () => cachedBaseUrl || LOCAL_URL;
