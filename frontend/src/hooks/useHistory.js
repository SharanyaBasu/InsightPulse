import { useEffect, useState } from "react";
import axios from "axios";

export default function useHistory() {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isActive = true;

    setError(null);
    axios
      .get("/api/history")
      .then((res) => {
        if (isActive) setHistory(res.data);
      })
      .catch((err) => {
        console.error("Failed to fetch history:", err);
        if (isActive) setError(err);
      })
      .finally(() => {
        if (isActive) setLoading(false);
      });

    return () => {
      isActive = false;
    };
  }, []);

  return { history, loading, error };
}
