import { useEffect } from "react";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid, ReferenceLine } from "recharts";

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0];
  const price = d.value;
  const formatted = price != null
    ? Number(price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : "—";

  return (
    <div style={{
      background: "#0a0a0a",
      border: "1px solid var(--panel-border)",
      padding: "0.4rem 0.7rem",
      fontSize: "0.78rem",
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      <div style={{ color: "var(--text-mute)", fontSize: "0.65rem", marginBottom: "0.15rem" }}>{label}</div>
      <div style={{ color: "var(--green)", fontWeight: 700, fontSize: "1rem" }}>{formatted}</div>
      {d.payload?.change != null && (
        <div style={{ color: d.payload.change >= 0 ? "var(--green)" : "var(--red)", fontSize: "0.72rem" }}>
          {d.payload.change >= 0 ? "+" : ""}{d.payload.change.toFixed(2)}%
        </div>
      )}
    </div>
  );
}

export default function ChartModal({ asset, data, onClose }) {
  useEffect(() => {
    function handleKey(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  if (!asset || !data || data.length === 0) return null;

  const firstPrice = data[0]?.price;
  const lastPrice = data[data.length - 1]?.price;
  const totalChange = firstPrice ? ((lastPrice - firstPrice) / firstPrice) * 100 : 0;
  const isUp = totalChange >= 0;
  const strokeColor = isUp ? "var(--green)" : "var(--red)";
  const fillColor = isUp ? "rgba(0, 255, 65, 0.06)" : "rgba(255, 51, 51, 0.06)";

  // Add change % to each data point
  const enriched = data.map((d) => ({
    ...d,
    change: firstPrice ? ((d.price - firstPrice) / firstPrice) * 100 : 0,
  }));

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.85)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 200,
        cursor: "pointer",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#0a0a0a",
          border: "1px solid var(--panel-border)",
          borderRadius: "var(--radius)",
          width: "min(90vw, 1100px)",
          maxHeight: "85vh",
          cursor: "default",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0.6rem 1rem",
          borderBottom: "1px solid var(--panel-border)",
        }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "1rem" }}>
            <span style={{ color: "var(--green)", fontWeight: 700, fontSize: "1rem", letterSpacing: "0.05em" }}>
              {asset.name || asset.symbol}
            </span>
            <span style={{ color: "var(--text)", fontWeight: 700, fontSize: "1.2rem" }}>
              {lastPrice != null ? Number(lastPrice).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"}
            </span>
            <span style={{ color: isUp ? "var(--green)" : "var(--red)", fontWeight: 600, fontSize: "0.85rem" }}>
              {isUp ? "+" : ""}{totalChange.toFixed(2)}% (30D)
            </span>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "1px solid var(--panel-border)",
              color: "var(--text-mute)",
              fontFamily: "inherit",
              fontSize: "0.7rem",
              padding: "0.2rem 0.5rem",
              cursor: "pointer",
              borderRadius: "var(--radius)",
            }}
          >
            ESC
          </button>
        </div>

        {/* Chart */}
        <div style={{ padding: "1rem", height: 420 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={enriched} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
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
              <ReferenceLine y={firstPrice} stroke="var(--text-mute)" strokeDasharray="4 4" strokeWidth={0.5} />
              <Area
                type="monotone"
                dataKey="price"
                stroke={strokeColor}
                fill={fillColor}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: strokeColor, stroke: "#0a0a0a", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
