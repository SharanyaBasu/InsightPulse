import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";

const REFRESH_INTERVAL = 5 * 60 * 1000;
const REQUEST_TIMEOUT = 180_000;

function isSummaryPayload(data) {
  if (!data || typeof data !== "object" || data.error) return false;
  if (data.headline) return true;
  return typeof data.summary === "string" && data.summary.trim().length > 0;
}

export default function useDailySummary() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const activeRequest = useRef(null);

  const fetchSummary = useCallback(async () => {
    if (activeRequest.current) return;
    const controller = new AbortController();
    activeRequest.current = controller;
    setLoading(true);
    setError(null);
    try {
      const deadline = Date.now() + REQUEST_TIMEOUT;
      let data;
      while (true) {
        const remaining = deadline - Date.now();
        if (remaining <= 0) throw new axios.AxiosError("Summary generation timed out. Please retry.", "ECONNABORTED");
        try {
          ({ data } = await axios.get("/api/summary/daily", {
            signal: controller.signal, timeout: remaining,
          }));
          break;
        } catch (err) {
          const retryAfter = Number(err.response?.headers?.["retry-after"]);
          if (err.response?.status !== 503 || !retryAfter) throw err;
          await new Promise((resolve, reject) => {
            const abort = () => {
              clearTimeout(timer);
              reject(new axios.CanceledError());
            };
            const timer = setTimeout(() => {
              controller.signal.removeEventListener("abort", abort);
              resolve();
            }, Math.min(retryAfter * 1000, Math.max(0, deadline - Date.now())));
            if (controller.signal.aborted) abort();
            else controller.signal.addEventListener("abort", abort, { once: true });
          });
        }
      }
      if (activeRequest.current !== controller) return;
      if (!isSummaryPayload(data)) {
        throw new Error(typeof data?.error === "string" ? data.error : "No valid market summary was returned.");
      }
      setSummary(data);
    } catch (err) {
      if (activeRequest.current !== controller || axios.isCancel(err)) return;
      const detail = err.response?.data?.detail || err.response?.data?.error;
      setError(err.code === "ECONNABORTED"
        ? "Summary generation timed out. Please retry."
        : typeof detail === "string" ? detail : err.message || "Failed to load summary.");
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchSummary();
    const id = setInterval(fetchSummary, REFRESH_INTERVAL);
    return () => {
      clearInterval(id);
      activeRequest.current?.abort();
      activeRequest.current = null;
    };
  }, [fetchSummary]);

  return { summary, loading, error, refresh: fetchSummary };
}
