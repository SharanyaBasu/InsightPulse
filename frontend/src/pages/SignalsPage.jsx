import { useState } from "react";
import { useMarketData } from "../context/MarketDataContext";
import useHistory from "../hooks/useHistory";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import TerminalPanel from "../components/Terminal/TerminalPanel";
import { AreaChart, Area, LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0a0a0a",
      border: "1px solid var(--panel-border)",
      padding: "0.4rem 0.7rem",
      fontSize: "0.78rem",
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      <div style={{ color: "var(--text-mute)", fontSize: "0.68rem", marginBottom: "0.15rem" }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontWeight: 600 }}>
          {p.name}: {typeof p.value === "number" ? Number(p.value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : p.value}
        </div>
      ))}
    </div>
  );
}

export default function SignalsPage() {
  const { overview } = useMarketData();
  const { history, loading } = useHistory();
  const [selectedAsset, setSelectedAsset] = useState("sp500");

  if (loading) return <TerminalLoader message="LOADING HISTORY" />;

  // History API returns: [{ date, sp500, nasdaq, gold, oil, ... }, ...]
  const rows = Array.isArray(history) ? history : [];

  const availableAssets = rows.length > 0
    ? Object.keys(rows[0]).filter((k) => k !== "date" && k !== "id")
    : [];

  const priceData = rows
    .filter((row) => row[selectedAsset] != null)
    .map((row) => ({ date: row.date, price: row[selectedAsset] }));

  // Multi-asset overlay
  const overlayData = rows.map((row) => ({
    date: row.date,
    sp500: row.sp500,
    vix: row.vix,
    gold: row.gold,
  }));

  return (
    <div>
      {/* Current sentiment summary */}
      {overview?.sentiment && (
        <TerminalPanel title="Current Sentiment" style={{ marginBottom: "0.5rem" }}>
          <div style={{ display: "flex", gap: "2.5rem", fontSize: "0.9rem" }}>
            <div>
              Label: <span style={{ color: overview.sentiment.score > 0 ? "var(--green)" : "var(--red)", fontWeight: 700 }}>
                {overview.sentiment.label}
              </span>
            </div>
            <div>
              Score: <span style={{ fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>
                {overview.sentiment.score?.toFixed(3)}
              </span>
            </div>
            <div>
              Trend: <span style={{ fontWeight: 600 }}>{overview.sentiment.equity_trend}</span>
            </div>
          </div>
        </TerminalPanel>
      )}

      {/* Price chart with asset selector */}
      <TerminalPanel
        title="Historical Prices"
        headerRight={
          <select
            value={selectedAsset}
            onChange={(e) => setSelectedAsset(e.target.value)}
            style={{
              background: "var(--bg)",
              color: "var(--text)",
              border: "1px solid var(--panel-border)",
              borderRadius: "var(--radius)",
              fontSize: "0.72rem",
              fontFamily: "inherit",
              padding: "0.2rem 0.4rem",
              outline: "none",
            }}
          >
            {availableAssets.map((a) => (
              <option key={a} value={a}>{a.toUpperCase()}</option>
            ))}
          </select>
        }
        style={{ marginBottom: "0.5rem" }}
      >
        {priceData.length > 0 ? (
          <div style={{ height: 340 }}>
            <ResponsiveContainer>
              <AreaChart data={priceData} margin={{ top: 10, right: 10, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: "#484848", fontFamily: "'JetBrains Mono', monospace" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--panel-border)" }}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "#484848", fontFamily: "'JetBrains Mono', monospace" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--panel-border)" }}
                  domain={["auto", "auto"]}
                  tickFormatter={(v) => v.toLocaleString()}
                />
                <Tooltip content={<ChartTooltip />} cursor={{ stroke: "var(--text-mute)", strokeDasharray: "3 3" }} />
                <Area
                  type="monotone"
                  dataKey="price"
                  stroke="var(--green)"
                  fill="rgba(0, 255, 65, 0.06)"
                  strokeWidth={1.8}
                  dot={false}
                  activeDot={{ r: 4, fill: "var(--green)", stroke: "#0a0a0a", strokeWidth: 2 }}
                  name={selectedAsset.toUpperCase()}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div style={{ color: "var(--text-mute)", fontSize: "0.85rem", padding: "2rem", textAlign: "center" }}>
            No history data for {selectedAsset.toUpperCase()}
          </div>
        )}
      </TerminalPanel>

      {/* Multi-asset comparison */}
      {overlayData.length > 0 && overlayData.some((d) => d.sp500 != null) && (
        <TerminalPanel title="Multi-Asset Overlay (SP500 / VIX / Gold)">
          <div style={{ height: 280 }}>
            <ResponsiveContainer>
              <LineChart data={overlayData} margin={{ top: 10, right: 10, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: "#484848", fontFamily: "'JetBrains Mono', monospace" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--panel-border)" }}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "#484848", fontFamily: "'JetBrains Mono', monospace" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--panel-border)" }}
                />
                <Tooltip content={<ChartTooltip />} cursor={{ stroke: "var(--text-mute)", strokeDasharray: "3 3" }} />
                <Line type="monotone" dataKey="sp500" stroke="var(--green)" dot={false} strokeWidth={1.5} name="SP500" activeDot={{ r: 3 }} />
                <Line type="monotone" dataKey="vix" stroke="var(--red)" dot={false} strokeWidth={1.5} name="VIX" activeDot={{ r: 3 }} />
                <Line type="monotone" dataKey="gold" stroke="var(--amber)" dot={false} strokeWidth={1.5} name="GOLD" activeDot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </TerminalPanel>
      )}
    </div>
  );
}
