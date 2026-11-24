import { useMarketData } from "../../context/useMarketData";

export default function BottomStatusBar() {
  const { overview, lastUpdated, loading, error } = useMarketData();

  if (loading || !overview) {
    return (
      <div className="bottom-status" style={barStyle}>
        <span style={{ color: "var(--text-mute)" }}>{error ? "DATA UNAVAILABLE" : "LOADING..."}</span>
      </div>
    );
  }

  const s = overview.sentiment || {};
  const scoreColor =
    s.score > 0 ? "var(--green)" : s.score < 0 ? "var(--red)" : "var(--text-soft)";

  const regime = overview.regime?.regime || "—";

  const updatedStr = lastUpdated
    ? lastUpdated.toLocaleString("en-US", { hour12: false })
    : "—";

  return (
    <div className="bottom-status" style={barStyle}>
      <div style={{ display: "flex", gap: "1.5rem", alignItems: "center" }}>
        <span>
          SENTIMENT:{" "}
          <span style={{ color: scoreColor, fontWeight: 600 }}>
            {s.label || "—"} {s.score != null ? (s.score >= 0 ? "+" : "") + s.score.toFixed(2) : ""}
          </span>
        </span>
        <span style={{ color: "var(--panel-border)" }}>|</span>
        <span>
          REGIME: <span style={{ color: "var(--amber)", fontWeight: 600 }}>{regime}</span>
        </span>
        <span style={{ color: "var(--panel-border)" }}>|</span>
        <span>
          TREND: <span style={{ fontWeight: 600 }}>{s.equity_trend || "—"}</span>
        </span>
      </div>
      <span style={{ color: "var(--text-mute)" }}>Last checked: {updatedStr}{error ? " · Refresh failed; showing previous data" : ""}</span>
    </div>
  );
}

const barStyle = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "0.3rem 0.8rem",
  background: "#080808",
  borderTop: "1px solid var(--panel-border)",
  fontSize: "0.7rem",
  color: "var(--text-soft)",
  minHeight: "26px",
};
