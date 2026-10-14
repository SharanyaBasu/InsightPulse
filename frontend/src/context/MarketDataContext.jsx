import { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";

import { MarketDataContext } from "./useMarketData";

const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes

export function MarketDataProvider({ children }) {
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const [refreshing, setRefreshing] = useState(false);
  const request = useRef(null);

  const fetchOverview = useCallback(() => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setRefreshing(true);
    axios
      .get("/api/overview", { signal: controller.signal, timeout: 30000 })
      .then((res) => {
        if (controller.signal.aborted) return;
        if (!res.data?.market_cards || !res.data?.macro) {
          throw new Error("Market snapshot unavailable");
        }
        setError(null);
        setOverview(res.data);
        setLastUpdated(new Date());
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(err);
      })
      .finally(() => { if (!controller.signal.aborted) { setLoading(false); setRefreshing(false); } });
  }, []);

  useEffect(() => {
    fetchOverview();
    const id = setInterval(fetchOverview, REFRESH_INTERVAL);
    return () => { clearInterval(id); request.current?.abort(); };
  }, [fetchOverview]);

  return (
    <MarketDataContext.Provider
      value={{ overview, loading, error, refreshing, lastUpdated, refresh: fetchOverview }}
    >
      {children}
    </MarketDataContext.Provider>
  );
}
