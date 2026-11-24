import { createContext, useContext } from "react";

export const MarketDataContext = createContext(null);

/** Access the shared MarketDataContext provider state. */
export function useMarketData() {
  const ctx = useContext(MarketDataContext);
  if (!ctx) throw new Error("useMarketData must be used within MarketDataProvider");
  return ctx;
}
