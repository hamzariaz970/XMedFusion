/**
 * API Configuration Utility
 * Handles fallback between Ngrok and Localhost
 */

const NGROK_URL = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_BASE_URL_NGROK;
const LOCAL_URL = import.meta.env.VITE_API_BASE_URL_LOCAL || "http://localhost:8000";
const LOOPBACK_URL = "http://127.0.0.1:8000";

let cachedBaseUrl: string | null = null;

/** Returns true when the cached/selected URL is the ngrok tunnel. */
export const isNgrokUrl = (url: string) =>
  url.includes("ngrok") || url.includes("ngrok-free");

/**
 * Returns headers required to bypass ngrok's browser interstitial page.
 * Without these, ngrok returns its own HTML (HTTP 200, no CORS headers)
 * instead of forwarding the request to FastAPI.
 * Should be included on ALL fetch calls when using a ngrok URL.
 */
export const getNgrokHeaders = (baseUrl?: string): Record<string, string> => {
  const url = baseUrl ?? cachedBaseUrl ?? "";
  if (isNgrokUrl(url)) {
    return { "ngrok-skip-browser-warning": "true" };
  }
  return {};
};

const getCandidateBaseUrls = () => {
  const candidates = [cachedBaseUrl, NGROK_URL, LOCAL_URL, LOOPBACK_URL]
    .filter(Boolean)
    .map((url) => (url as string).replace(/\/$/, ""));

  return Array.from(new Set(candidates));
};

/**
 * Probes whether an API base URL is reachable.
 *
 * Uses `no-cors` mode so the browser doesn't block the OPTIONS preflight,
 * which lets us detect connectivity. The real CORS check is enforced by the
 * backend on all subsequent credentialed requests (FastAPI CORSMiddleware).
 */
const canReachApi = async (baseUrl: string, timeoutMs = 2500) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    // Query param + header together: the query param skips the browser
    // warning page for GET requests; the header ensures ngrok forwards
    // the request to FastAPI instead of returning its own interstitial.
    const url = `${baseUrl}/api/health?ngrok-skip-browser-warning=1`;

    const response = await fetch(url, {
      method: "GET",
      signal: controller.signal,
      mode: "no-cors", // Intentional: we only care about reachability here
      headers: getNgrokHeaders(baseUrl),
    });

    clearTimeout(timeoutId);
    // In no-cors mode the response type is "opaque" (status=0).
    // If we didn't throw, the server is reachable.
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
      if (isNgrokUrl(candidateUrl)) {
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
 * otherwise returns local as a safe default while the async check happens.
 */
export const getApiBaseSync = () => cachedBaseUrl || LOCAL_URL;

