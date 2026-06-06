import { useCallback, useEffect, useState } from "react";
import axios from "axios";

const REFRESH_INTERVAL = 5 * 60 * 1000;

function isSummaryPayload(data) {
  if (!data || typeof data !== "object" || data.error) return false;
  if (data.headline) return true;
  return typeof data.summary === "string" && data.summary.trim().length > 0;
}

export default function useDailySummary() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchSummary = useCallback(() => {
    setLoading(true);
    setError(null);

    axios
      .get("/api/summary/daily")
      .then((res) => {
        const data = res.data;
        if (data?.error) {
          setSummary(null);
          setError(typeof data.error === "string" ? data.error : "Summary unavailable.");
          return;
        }
        if (!isSummaryPayload(data)) {
          setSummary(null);
          setError(null);
          return;
        }
        setSummary(data);
        setError(null);
      })
      .catch((err) => {
        console.error("Failed to fetch daily summary:", err);
        setSummary(null);
        setError(err.response?.data?.error || err.message || "Failed to load summary.");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchSummary();
    const id = setInterval(fetchSummary, REFRESH_INTERVAL);
    return () => clearInterval(id);
  }, [fetchSummary]);

  return { summary, loading, error, refresh: fetchSummary };
}
