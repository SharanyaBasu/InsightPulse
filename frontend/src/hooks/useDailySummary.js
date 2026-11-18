import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";

const REFRESH_INTERVAL = 5 * 60 * 1000;
const POLL_INTERVAL = 3_000;
const REQUEST_TIMEOUT = 15_000;

function isSummaryPayload(data) {
  if (!data || typeof data !== "object" || data.error) return false;
  if (data.headline || data.refresh) return true;
  return typeof data.summary === "string" && data.summary.trim().length > 0;
}

export default function useDailySummary() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const activeRequest = useRef(null);
  const pollTimer = useRef(null);
  const hasSummary = useRef(false);

  const fetchSummary = useCallback(async (enqueue = false) => {
    if (activeRequest.current) return;
    const controller = new AbortController();
    activeRequest.current = controller;
    if (enqueue) setPending(true);
    if (!hasSummary.current) setLoading(true);
    try {
      const { data } = await axios.get("/api/summary/daily", {
        params: enqueue ? { refresh_data: true } : undefined,
        signal: controller.signal,
        timeout: REQUEST_TIMEOUT,
      });
      if (activeRequest.current !== controller) return;
      if (!isSummaryPayload(data)) {
        throw new Error(typeof data?.error === "string" ? data.error : "No valid market summary was returned.");
      }
      hasSummary.current = Boolean(data.headline);
      setSummary(data);
      setError(null);
      if (pollTimer.current) clearTimeout(pollTimer.current);
      if (data.refresh?.in_progress) {
        pollTimer.current = setTimeout(() => {
          pollTimer.current = null;
          fetchSummary(false);
        }, POLL_INTERVAL);
      }
    } catch (err) {
      if (activeRequest.current !== controller || axios.isCancel(err)) return;
      const detail = err.response?.data?.detail || err.response?.data?.error;
      setError(err.code === "ECONNABORTED"
        ? "The saved snapshot took too long to load. Please retry."
        : typeof detail === "string" ? detail : err.message || "Failed to load summary.");
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        setLoading(false);
        setPending(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchSummary(false);
    const id = setInterval(() => fetchSummary(false), REFRESH_INTERVAL);
    return () => {
      clearInterval(id);
      if (pollTimer.current) clearTimeout(pollTimer.current);
      activeRequest.current?.abort();
      activeRequest.current = null;
    };
  }, [fetchSummary]);

  const refresh = useCallback(() => fetchSummary(true), [fetchSummary]);
  return { summary, loading, pending, error, refresh };
}
