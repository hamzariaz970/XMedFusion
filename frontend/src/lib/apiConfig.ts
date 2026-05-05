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

    // Using query param instead of header avoids CORS preflight OPTIONS requests 
    // which Ngrok sometimes blocks on free accounts.
    const url = `${baseUrl}/api/health?ngrok-skip-browser-warning=1`;
    
    const response = await fetch(url, {
      method: "GET",
      signal: controller.signal,
      mode: "no-cors", // Bypasses CORS blocks for the reachability check
    });

    clearTimeout(timeoutId);
    // In no-cors mode, status is 0. If we got here without an exception, it means 
    // the server responded (even if it was a CORS-restricted response).
    return response.type === "opaque" || response.ok;
  } catch (err) {
    console.warn(`Connection to ${baseUrl} failed:`, err);
    return false;
  }
};

export const getApiBase = async (forceRefresh = false): Promise<string> => {
  if (cachedBaseUrl && !forceRefresh && await canReachApi(cachedBaseUrl, 1200)) {
    return cachedBaseUrl;
  }

  const candidates = getCandidateBaseUrls();
  console.log("🔍 API Discovery - Candidates:", candidates);
  
  if (!NGROK_URL) {
    console.warn("⚠️ VITE_API_BASE_URL is missing from environment variables!");
  }

  for (const candidateUrl of candidates) {
    console.log(`Testing connectivity to: ${candidateUrl}...`);
    if (await canReachApi(candidateUrl)) {
      if (candidateUrl === NGROK_URL) {
        console.log("🚀 SUCCESS: Using Remote Ngrok API:", candidateUrl);
      } else {
        console.log("🏠 SUCCESS: Using Local API:", candidateUrl);
      }
      cachedBaseUrl = candidateUrl;
      return candidateUrl;
    }
  }

  console.error("❌ FAILURE: No reachable API found. Falling back to default:", LOCAL_URL);
  cachedBaseUrl = LOCAL_URL;
  return LOCAL_URL;
};

/**
 * Helper to get the base URL synchronously if already cached, 
 * otherwise returns local as a safe default while the check happens.
 */
export const getApiBaseSync = () => cachedBaseUrl || LOCAL_URL;
