import useHistory from "./useHistory";
export default function useAssetHistory(symbol) {
  const { history, ...state } = useHistory(Boolean(symbol));
  const data = (history || []).filter(row => typeof row[symbol] === "number" && Number.isFinite(row[symbol]))
    .map(row => ({ date: row.date, price: row[symbol] }));
  return { data, ...state };
}
