import { useState } from "react";
import { Treemap, ResponsiveContainer } from "recharts";

// Interpolates between two hex colors by t (0–1)
function lerpColor(a, b, t) {
  const ah = parseInt(a.slice(1), 16);
  const bh = parseInt(b.slice(1), 16);
  const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
  const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const b_ = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${b_})`;
}

function getTileColor(change, maxAbs) {
  const t = Math.min(Math.abs(change) / maxAbs, 1);
  if (change >= 0) return lerpColor("#1a3d2b", "#1dd75f", t);
  return lerpColor("#3d1a1a", "#ff4e4e", t);
}

function CustomContent({ x, y, width, height, name, change, color }) {
  if (width < 20 || height < 20) return null;
  if (change === undefined || change === null) return null;

  const fontSize = Math.min(width / 7, height / 4, 18);
  const smallFont = Math.max(fontSize * 0.7, 9);
  const showName = height > 36 && width > 40;
  const sign = change >= 0 ? "+" : "";

  return (
    <g>
      <rect
        x={x + 1}
        y={y + 1}
        width={width - 2}
        height={height - 2}
        style={{ fill: color, stroke: "#0b0e11", strokeWidth: 2 }}
        rx={4}
      />
      {showName && (
        <text
          x={x + width / 2}
          y={y + height / 2 - (height > 60 ? fontSize * 0.7 : 0)}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="#ffffff"
          fontSize={fontSize}
          fontWeight={700}
          fontFamily="Inter, sans-serif"
        >
          {name}
        </text>
      )}
      {height > 60 && (
        <text
          x={x + width / 2}
          y={y + height / 2 + fontSize * 0.9}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgba(255,255,255,0.85)"
          fontSize={smallFont}
          fontWeight={500}
          fontFamily="Inter, sans-serif"
        >
          {sign}{change.toFixed(2)}%
        </text>
      )}
      {height <= 60 && height > 36 && (
        <text
          x={x + width / 2}
          y={y + height / 2}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgba(255,255,255,0.85)"
          fontSize={Math.min(smallFont, 11)}
          fontWeight={600}
          fontFamily="Inter, sans-serif"
        >
          {showName ? `${sign}${change.toFixed(2)}%` : name}
        </text>
      )}
    </g>
  );
}

function InfoTooltip() {
  const [visible, setVisible] = useState(false);

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      {/* "?" button */}
      <button
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        onClick={() => setVisible((v) => !v)}
        style={{
          width: 18,
          height: 18,
          borderRadius: "50%",
          border: "1px solid var(--text-soft)",
          background: "transparent",
          color: "var(--text-soft)",
          fontSize: 11,
          fontWeight: 700,
          cursor: "pointer",
          lineHeight: 1,
          padding: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        ?
      </button>

      {/* Tooltip */}
      {visible && (
        <div
          style={{
            position: "absolute",
            top: 24,
            left: 0,
            zIndex: 100,
            width: 280,
            background: "#1b1e27",
            border: "1px solid var(--panel-border)",
            borderRadius: 8,
            padding: "0.85rem 1rem",
            boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
            color: "var(--text-soft)",
            fontSize: "0.78rem",
            lineHeight: 1.6,
          }}
        >
          <p style={{ margin: "0 0 0.5rem", color: "var(--text)", fontWeight: 600, fontSize: "0.82rem" }}>
            How are these percentages calculated?
          </p>
          <p style={{ margin: "0 0 0.5rem" }}>
            Each sector's projected change is computed from real day-over-day market
            signal shocks — NASDAQ, S&P 500, oil, 10Y yield, USD index, and copper.
          </p>
          <p style={{ margin: "0 0 0.5rem" }}>
            Each signal is weighted based on how sensitive that sector historically
            is to it. For example, Energy is heavily weighted to oil, while Utilities
            are negatively weighted to interest rates.
          </p>
          <p style={{ margin: 0, color: "var(--text-mute)", fontSize: "0.74rem" }}>
            Tile size reflects the sector's share of the S&P 500. Color intensity
            reflects the magnitude of the projected move.
          </p>
        </div>
      )}
    </div>
  );
}

export default function SectorHeatmap({ sectors }) {
  if (!sectors?.length) return null;

  const maxAbs = Math.max(...sectors.map((s) => Math.abs(s.change)));

  const data = sectors.map((s) => ({
    ...s,
    change: s.change ?? 0,
    size: s.weight ?? Math.max(Math.abs(s.change ?? 0), 0.1),
    color: getTileColor(s.change ?? 0, maxAbs),
  }));

  return (
    <div>
      {/* Header row with title and info icon */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
        <h3 style={{ color: "var(--blue)", margin: 0 }}>
          Sector Heatmap
        </h3>
        <InfoTooltip />
      </div>

      {/* Legend */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
        <span style={{ fontSize: "0.75rem", color: "var(--text-soft)" }}>Negative</span>
        <div style={{
          height: "8px",
          width: "140px",
          borderRadius: "4px",
          background: "linear-gradient(to right, #ff4e4e, #1a3d2b, #1dd75f)",
        }} />
        <span style={{ fontSize: "0.75rem", color: "var(--text-soft)" }}>Positive</span>
      </div>

      <div style={{ width: "100%", height: 420, minHeight: 420 }}>
        <ResponsiveContainer width="100%" height="100%">
          <Treemap
            data={data}
            dataKey="size"
            aspectRatio={4 / 3}
            content={(props) => <CustomContent {...props} />}
          />
        </ResponsiveContainer>
      </div>
    </div>
  );
}
