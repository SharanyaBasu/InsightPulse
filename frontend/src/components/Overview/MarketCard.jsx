import React from "react";

export default function MarketCard({ card }) {
  const change = card.change_1d;
  const isUp = change > 0;
  const isDown = change < 0;

  const changeColor = isUp
    ? "var(--green)"
    : isDown
    ? "var(--red)"
    : "var(--text-soft)";

  return (
    <div
      style={{
        background: "rgba(28, 33, 40, 0.85)",
        backdropFilter: "blur(6px)",
        borderRadius: "12px",
        padding: "1.1rem 1.2rem",
        border: "1px solid var(--border)",
        boxShadow:
          "0 0 14px rgba(0, 0, 0, 0.35), inset 0 0 0 1px rgba(255,255,255,0.03)",
        transition: "transform 0.15s ease, box-shadow 0.15s ease",
      }}
      className="market-card"
    >
      {/* TITLE */}
      <div
        style={{
          fontSize: "0.85rem",
          color: "var(--text-soft)",
          marginBottom: "0.35rem",
          letterSpacing: "0.3px",
          fontWeight: 500,
        }}
      >
        {card.name}
      </div>

      {/* PRICE */}
      <div
        style={{
          fontSize: "1.45rem",
          fontWeight: 700,
          marginBottom: "0.15rem",
          color: "var(--text)",
        }}
      >
        {Number(card.price).toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}
      </div>

      {/* DAILY CHANGE */}
      <div
        style={{
          fontSize: "0.95rem",
          fontWeight: 600,
          color: changeColor,
          marginBottom: "0.7rem",
        }}
      >
        {isUp && "▲ "}
        {isDown && "▼ "}
        {change.toFixed(2)}%
      </div>

      {/* SPARKLINE */}
      <div
        style={{
          height: "38px",
          width: "100%",
          borderRadius: "6px",
          background:
            "linear-gradient(90deg, rgba(74,168,255,0.25), rgba(74,168,255,0))",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <svg
          width="100%"
          height="38"
          viewBox="0 0 100 38"
          preserveAspectRatio="none"
        >
          <polyline
            fill="none"
            stroke="#4aa8ff"
            strokeWidth="1.5"
            points={card.sparkline
              .map((v, i) => `${(i / card.sparkline.length) * 100},${38 - v * 30}`)
              .join(" ")}
          />
        </svg>
      </div>
    </div>
  );
}
