import { useMarketData } from "../context/MarketDataContext";

export default function useOverview() {
  const { overview, loading } = useMarketData();
  return { data: overview, loading };
}
