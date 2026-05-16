export default function MarketCard({ card, onClick }) {
  const change = card.change_1d;
  const isUp = change > 0;
  const isDown = change < 0;
  const changeColor = isUp ? "var(--green)" : isDown ? "var(--red)" : "var(--text-soft)";

  return (
    <div
      onClick={onClick}
      style={{
        background: "var(--panel)",
        border: "1px solid var(--panel-border)",
        borderRadius: "var(--radius)",
        padding: "0.6rem 0.8rem",
        cursor: onClick ? "pointer" : "default",
        transition: "border-color 0.15s",
      }}
      onMouseEnter={(e) => onClick && (e.currentTarget.style.borderColor = "var(--text-mute)")}
      onMouseLeave={(e) => onClick && (e.currentTarget.style.borderColor = "var(--panel-border)")}
    >
      {/* Symbol */}
      <div style={{ fontSize: "0.72rem", color: "var(--text-mute)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "0.15rem" }}>
        {card.name}
      </div>

      {/* Price */}
      <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text)", marginBottom: "0.2rem" }}>
        {Number(card.price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      </div>

      {/* Change + Sparkline row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.5rem" }}>
        <span style={{ fontSize: "0.88rem", fontWeight: 600, color: changeColor }}>
          {isUp ? "+" : ""}{change.toFixed(2)}%
        </span>
        <svg width="70" height="28" viewBox="0 0 70 28" preserveAspectRatio="none" style={{ flexShrink: 0 }}>
          <polyline
            fill="none"
            stroke={changeColor}
            strokeWidth="1.5"
            points={card.sparkline
              .map((v, i) => `${(i / card.sparkline.length) * 70},${28 - v * 22}`)
              .join(" ")}
          />
        </svg>
      </div>

      {/* Extra changes if available */}
      {(card.change_1w != null || card.change_1m != null) && (
        <div style={{ display: "flex", gap: "0.8rem", marginTop: "0.3rem", fontSize: "0.72rem", color: "var(--text-mute)" }}>
          {card.change_1w != null && (
            <span>
              1W: <span style={{ color: card.change_1w >= 0 ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                {card.change_1w >= 0 ? "+" : ""}{card.change_1w.toFixed(2)}%
              </span>
            </span>
          )}
          {card.change_1m != null && (
            <span>
              1M: <span style={{ color: card.change_1m >= 0 ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                {card.change_1m >= 0 ? "+" : ""}{card.change_1m.toFixed(2)}%
              </span>
            </span>
          )}
        </div>
      )}
    </div>
  );
}
