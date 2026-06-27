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
  if (change === undefined || change === null) return null; // ADD THIS

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

export default function SectorHeatmap({ sectors }) {
  if (!sectors?.length) return null;

  const maxAbs = Math.max(...sectors.map((s) => Math.abs(s.change)));

  // `size` drives tile area — use weight if available, else fall back to abs(change)
  const data = sectors.map((s) => ({
  ...s,
  change: s.change ?? 0,
  size: s.weight ?? Math.max(Math.abs(s.change ?? 0), 0.1),
  color: getTileColor(s.change ?? 0, maxAbs),
  }));

  return (
    <div>
      <h3 style={{ color: "var(--blue)", marginBottom: "1rem", marginTop: 0 }}>
        Sector Heatmap
      </h3>

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

      <div style={{ width: "100%", height: 420 }}>
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
