import { useState, useEffect } from "react";
import { connectionStatus, observationRange } from "../../context/dataFreshness";
import { useMarketData } from "../../context/useMarketData";

function getMarketSession() {
  const now = new Date();
  const ny = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  const h = ny.getHours();
  const m = ny.getMinutes();
  const day = ny.getDay();
  const t = h * 60 + m;

  if (day === 0 || day === 6) return { label: "CLOSED", color: "var(--text-mute)" };
  if (t < 570) return { label: "PRE-MARKET", color: "var(--amber)" };
  if (t < 960) return { label: "MARKET OPEN", color: "var(--green)" };
  if (t < 1200) return { label: "AFTER-HOURS", color: "var(--amber)" };
  return { label: "CLOSED", color: "var(--text-mute)" };
}

export default function TopStatusBar() {
  const [time, setTime] = useState(new Date());
  const { error, overview, loading } = useMarketData();

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const session = getMarketSession();
  const status = connectionStatus({ error, overview, loading });
  const connColor = error ? "var(--red)" : overview ? "var(--text-soft)" : "var(--amber)";

  return (
    <div className="top-status"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0.35rem 0.8rem",
        background: "#080808",
        borderBottom: "1px solid var(--panel-border)",
        fontSize: "0.75rem",
        minHeight: "28px",
      }}
    >
      {/* Left: Brand */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <span style={{ color: "var(--green)", fontWeight: 700, letterSpacing: "0.1em" }}>
          INSIGHTPULSE
        </span>
        <span className="t-blink" style={{ color: "var(--green)" }}>_</span>
      </div>

      {/* Center: Market status */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem", color: "var(--text-mute)" }}>
        <span>MARKET ANALYTICS TERMINAL</span>
        <span style={{ color: session.color }} title="Typical US market hours; holidays are not reflected">{session.label}</span>
        <span>Market data as of {observationRange(overview?.data_dates?.market)}</span>
      </div>

      {/* Right: Clock + connection */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.8rem" }}>
        <span style={{ color: "var(--text-soft)" }}>
          {time.toLocaleTimeString("en-US", { hour12: false })}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
          <div
            title="API connection status; not a live-data indicator"
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: connColor,
            }}
          />
          <span style={{ color: connColor, fontSize: "0.65rem" }}>
            {status}
          </span>
        </div>
      </div>
    </div>
  );
}
