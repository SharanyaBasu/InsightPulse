export const CHART_WINDOWS = [{ label: "1M", value: 21 }, { label: "3M", value: 63 }, { label: "1Y", value: 252 }, { label: "All", value: 0 }];
export function windowRows(rows, count) { return count ? rows.slice(-count) : rows; }
export function normalizeComparison(rows, keys) {
  const shared = rows.filter(row => keys.every(key => typeof row[key] === "number" && Number.isFinite(row[key]) && row[key] > 0));
  if (!keys.length || shared.length < 2) return [];
  return shared.map(row => Object.fromEntries([["date", row.date], ...keys.map(key => [key, (row[key] / shared[0][key] - 1) * 100])]));
}
export function assetUnit(key) {
  if (["irx", "fvx", "tnx", "dgs10", "dgs2"].includes(key)) return "% yield";
  if (["sp500", "nasdaq", "vix", "usd_index"].includes(key)) return "index points";
  if (["eurusd", "gbpusd", "audusd"].includes(key)) return "USD per unit";
  if (key === "usdjpy") return "JPY per USD";
  if (key === "usdchf") return "CHF per USD";
  if (key === "gold") return "USD / troy oz";
  if (key === "oil") return "USD / barrel";
  if (key === "copper") return "USD / lb";
  if (["bitcoin", "ethereum"].includes(key)) return "USD / coin";
  if (["vgk", "ewj", "eem", "cew", "tlt", "xlk", "xlf", "xly", "xlp", "xle", "xlv", "xli", "xlb", "xlre", "xlc"].includes(key)) return "USD / share";
  return "quote units";
}
export const shortDate = value => new Date(`${String(value).slice(0, 10)}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });
export const numberLabel = value => Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
