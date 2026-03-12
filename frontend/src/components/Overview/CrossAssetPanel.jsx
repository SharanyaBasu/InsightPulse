import React from "react";

const GROUPS = ["Equities", "Rates", "FX", "Commodities", "Crypto"];

function SparklineSVG({ data, change1d }) {
  if (!data || data.length < 2) return null;

  const stroke =
    change1d > 0 ? "var(--green)" : change1d < 0 ? "var(--red)" : "#4aa8ff";

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 60;
  const h = 24;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 2) - 1;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      <polyline fill="none" stroke={stroke} strokeWidth="1.5" points={points} />
    </svg>
  );
}

function fmt(price) {
  if (price == null) return "—";
  return Number(price).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function ChangeCell({ change1d }) {
  if (change1d == null) return <span style={{ color: "var(--text-soft)" }}>—</span>;
  const isUp = change1d > 0;
  const isDown = change1d < 0;
  const color = isUp ? "var(--green)" : isDown ? "var(--red)" : "var(--text-soft)";
  const arrow = isUp ? "▲" : isDown ? "▼" : "→";
  return (
    <span style={{ color, fontSize: "0.88rem" }}>
      {arrow} {Math.abs(change1d).toFixed(2)}%
    </span>
  );
}

export default function CrossAssetPanel({ assets }) {
  if (!assets || assets.length === 0) return null;

  const grouped = GROUPS.reduce((acc, g) => {
    const rows = assets.filter((a) => a.group === g);
    if (rows.length > 0) acc[g] = rows;
    return acc;
  }, {});

  return (
    <div
      style={{
        background: "var(--panel)",
        borderRadius: "12px",
        border: "1px solid var(--panel-border)",
        boxShadow: "0 0 10px var(--panel-glow)",
        padding: "1.4rem 1.6rem",
      }}
    >
      {Object.entries(grouped).map(([group, rows], gi) => (
        <div key={group}>
          <div
            style={{
              fontSize: "0.7rem",
              letterSpacing: "0.12em",
              color: "var(--text-mute)",
              textTransform: "uppercase",
              marginBottom: "0.5rem",
              marginTop: gi === 0 ? 0 : "1.2rem",
            }}
          >
            {group}
          </div>
          {rows.map((asset) => (
            <div
              key={asset.symbol || asset.name}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "0.5rem 0",
                borderBottom: "1px solid rgba(255,255,255,0.04)",
              }}
            >
              <span
                style={{
                  color: "var(--text-soft)",
                  fontSize: "0.88rem",
                  minWidth: "160px",
                }}
              >
                {asset.name}
              </span>
              <span
                style={{
                  color: "var(--text)",
                  fontSize: "0.95rem",
                  fontWeight: 600,
                  minWidth: "90px",
                  textAlign: "right",
                }}
              >
                {fmt(asset.price)}
              </span>
              <span style={{ minWidth: "70px", textAlign: "right" }}>
                <ChangeCell change1d={asset.change_1d} />
              </span>
              <div style={{ width: "60px", display: "flex", justifyContent: "flex-end" }}>
                <SparklineSVG data={asset.sparkline} change1d={asset.change_1d} />
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
