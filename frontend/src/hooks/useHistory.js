import { useEffect, useState, useCallback } from "react";
import axios from "axios";

export default function useHistory(enabled = true) {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState(null);
  const [attempt, setAttempt] = useState(0);
  const refresh = useCallback(() => setAttempt(value => value + 1), []);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    axios.get("/api/history", { signal: controller.signal, timeout: 30000 })
      .then(({ data }) => {
        if (!Array.isArray(data)) throw new Error("History response unavailable");
        if (!controller.signal.aborted) setHistory([...data].sort((a, b) => String(a.date).localeCompare(String(b.date))));
      })
      .catch(err => { if (!controller.signal.aborted) setError(err); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [enabled, attempt]);
  return { history, loading, error, refresh };
}
