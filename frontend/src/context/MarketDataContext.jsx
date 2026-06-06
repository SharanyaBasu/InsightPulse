import { createContext, useContext, useState, useEffect, useCallback } from "react";
import axios from "axios";

const MarketDataContext = createContext(null);

const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes

export function MarketDataProvider({ children }) {
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchOverview = useCallback(() => {
    setError(null);
    axios
      .get("/api/overview")
      .then((res) => {
        setOverview(res.data);
        setLastUpdated(new Date());
      })
      .catch((err) => {
        console.error("Failed to fetch overview:", err);
        setError(err);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchOverview();
    const id = setInterval(fetchOverview, REFRESH_INTERVAL);
    return () => clearInterval(id);
  }, [fetchOverview]);

  return (
    <MarketDataContext.Provider
      value={{ overview, loading, error, lastUpdated, refresh: fetchOverview }}
    >
      {children}
    </MarketDataContext.Provider>
  );
}

export function useMarketData() {
  const ctx = useContext(MarketDataContext);
  if (!ctx) throw new Error("useMarketData must be used within MarketDataProvider");
  return ctx;
}
