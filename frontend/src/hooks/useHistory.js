import { useEffect, useState } from "react";
import axios from "axios";

export default function useHistory() {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get("/api/history")
      .then((res) => setHistory(res.data))
      .catch((err) => console.error("Failed to fetch history:", err))
      .finally(() => setLoading(false));
  }, []);

  return { history, loading };
}
