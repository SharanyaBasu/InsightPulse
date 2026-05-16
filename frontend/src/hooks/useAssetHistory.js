import { useEffect, useState } from "react";
import axios from "axios";

// Fetches /api/history and returns price series for a given symbol
// History API returns: [{ date, sp500, nasdaq, gold, ... }, ...]
let cachedHistory = null;

export default function useAssetHistory(symbol) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!symbol) return;

    setLoading(true);

    const resolve = (hist) => {
      if (!hist || !Array.isArray(hist)) {
        setData(null);
        setLoading(false);
        return;
      }
      const series = hist
        .filter((row) => row[symbol] != null)
        .map((row) => ({ date: row.date, price: row[symbol] }));
      setData(series);
      setLoading(false);
    };

    if (cachedHistory) {
      resolve(cachedHistory);
    } else {
      axios
        .get("/api/history")
        .then((res) => {
          cachedHistory = res.data;
          resolve(cachedHistory);
        })
        .catch(() => {
          setData(null);
          setLoading(false);
        });
    }
  }, [symbol]);

  return { data, loading };
}
