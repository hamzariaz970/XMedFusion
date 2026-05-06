/**
 * API Configuration Utility
 * Handles fallback between Ngrok and Localhost
 */

const NGROK_URL = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_BASE_URL_NGROK;
const LOOPBACK_URL = "http://127.0.0.1:8000";
const LOCAL_URL = import.meta.env.VITE_API_BASE_URL_LOCAL || LOOPBACK_URL;

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

const isFrontendOnLocalhost = () => {
  if (typeof window === "undefined") {
    return false;
  }

  return ["localhost", "127.0.0.1"].includes(window.location.hostname);
};

const getCandidateBaseUrls = () => {
  const preferredUrls = isFrontendOnLocalhost()
    ? [LOOPBACK_URL, LOCAL_URL, cachedBaseUrl, NGROK_URL]
    : [cachedBaseUrl, NGROK_URL];

  const candidates = preferredUrls
    .filter(Boolean)
    .map((url) => (url as string).replace(/\/$/, ""));

  return Array.from(new Set(candidates));
};

/**
 * Probes whether an API base URL is the actual FastAPI backend.
 * A generic reachable page is not enough here: stopped ngrok tunnels can still
 * return HTML/error pages that would break upload requests later.
 */
const canReachApi = async (baseUrl: string, timeoutMs = isNgrokUrl(baseUrl) ? 8000 : 2500) => {
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
      headers: getNgrokHeaders(baseUrl),
    });

    clearTimeout(timeoutId);
    if (!response.ok) return false;

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) return false;

    const data = await response.json();
    return data?.status === "healthy" || data?.status === "degraded";
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

