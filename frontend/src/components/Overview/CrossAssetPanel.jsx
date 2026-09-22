const GROUPS = ["Equities", "Rates", "FX", "Commodities", "Crypto"];

function SparklineSVG({ data, change1d }) {
  if (!data || data.length < 2) return null;
  const stroke = change1d > 0 ? "var(--green)" : change1d < 0 ? "var(--red)" : "var(--text-mute)";
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 60;
  const h = 22;
  const points = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * (h - 2) - 1}`)
    .join(" ");

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      <polyline fill="none" stroke={stroke} strokeWidth="1.3" points={points} />
    </svg>
  );
}

function fmt(price) {
  if (price == null) return "—";
  return Number(price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function ChangeCell({ value }) {
  if (value == null) return <span style={{ color: "var(--text-mute)" }}>—</span>;
  const color = value > 0 ? "var(--green)" : value < 0 ? "var(--red)" : "var(--text-mute)";
  return (
    <span style={{ color, fontSize: "0.82rem", fontVariantNumeric: "tabular-nums" }}>
      {value > 0 ? "+" : ""}{Math.abs(value).toFixed(2)}%
    </span>
  );
}

export default function CrossAssetPanel({ assets, onRowClick }) {
  if (!assets || assets.length === 0) return null;

  const grouped = GROUPS.reduce((acc, g) => {
    const rows = assets.filter((a) => a.group === g);
    if (rows.length > 0) acc[g] = rows;
    return acc;
  }, {});

  return (
    <div style={{ background: "var(--panel)", border: "1px solid var(--panel-border)", borderRadius: "var(--radius)" }}>
      {/* Header row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(140px, 2fr) minmax(90px, 1fr) minmax(80px, 1fr) 60px",
          padding: "0.4rem 1rem",
          borderBottom: "1px solid var(--panel-border)",
          fontSize: "0.7rem",
          fontWeight: 600,
          color: "var(--text-mute)",
          textTransform: "uppercase",
          letterSpacing: "0.1em",
        }}
      >
        <span>Name</span>
        <span style={{ textAlign: "right" }}>Price</span>
        <span style={{ textAlign: "right" }}>1D Chg</span>
        <span style={{ textAlign: "right" }}>30D</span>
      </div>

      {Object.entries(grouped).map(([group, rows], gi) => (
        <div key={group}>
          {/* Group header */}
          <div
            style={{
              fontSize: "0.68rem",
              letterSpacing: "0.12em",
              color: "var(--green)",
              textTransform: "uppercase",
              padding: "0.5rem 1rem 0.25rem",
              marginTop: gi === 0 ? 0 : "0.15rem",
              fontWeight: 700,
            }}
          >
            {group}
          </div>
          {rows.map((asset) => (
            <div
              key={asset.symbol || asset.name}
              onClick={() => onRowClick?.(asset)}
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(140px, 2fr) minmax(90px, 1fr) minmax(80px, 1fr) 60px",
                alignItems: "center",
                padding: "0.35rem 1rem",
                borderBottom: "1px solid rgba(255,255,255,0.025)",
                fontSize: "0.85rem",
                cursor: onRowClick ? "pointer" : "default",
                transition: "background 0.1s",
              }}
              onMouseEnter={(e) => onRowClick && (e.currentTarget.style.background = "var(--panel-hover)")}
              onMouseLeave={(e) => onRowClick && (e.currentTarget.style.background = "transparent")}
            >
              <span style={{ color: "var(--text)", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {asset.name}
              </span>
              <span style={{ textAlign: "right", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                {fmt(asset.price)}
              </span>
              <span style={{ textAlign: "right" }}>
                <ChangeCell value={asset.change_1d} />
              </span>
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <SparklineSVG data={asset.sparkline} change1d={asset.change_1d} />
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
