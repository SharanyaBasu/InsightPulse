import React from "react";
import Sparkline from "./Sparkline";

function labelStyle(label) {
  if (label === "Positive")
    return { bg: "rgba(29,215,95,0.12)", color: "var(--green)" };
  if (label === "Negative")
    return { bg: "rgba(255,78,78,0.12)", color: "var(--red)" };
  return { bg: "rgba(156,163,175,0.1)", color: "var(--text-soft)" };
}

function corrColor(corr) {
  if (corr == null) return "var(--text-soft)";
  if (corr >= 0.3) return "var(--green)";
  if (corr <= -0.3) return "var(--red)";
  return "var(--text-soft)";
}

function GaugeBar({ corr, label }) {
  if (corr == null) return null;
  const { color } = labelStyle(label);
  const left = corr >= 0 ? "50%" : `${50 + corr * 50}%`;
  const width = `${Math.abs(corr) * 50}%`;

  return (
    <div
      style={{
        height: "4px",
        borderRadius: "4px",
        background: "rgba(255,255,255,0.07)",
        position: "relative",
        margin: "0.5rem 0",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          height: "100%",
          left,
          width,
          background: color,
          borderRadius: "4px",
        }}
      />
    </div>
  );
}

function CorrelationCard({ pair }) {
  const { color: labelColor, bg: labelBg } = labelStyle(pair.label);

  return (
    <div
      style={{
        background: "var(--panel)",
        borderRadius: "12px",
        border: "1px solid var(--panel-border)",
        boxShadow: "0 0 10px var(--panel-glow)",
        padding: "1.2rem 1.4rem",
      }}
    >
      <div
        style={{
          fontSize: "0.8rem",
          color: "var(--text-mute)",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          marginBottom: "0.6rem",
        }}
      >
        {pair.pair}
      </div>

      <div
        style={{
          fontSize: "2.2rem",
          fontWeight: 700,
          color: corrColor(pair.correlation),
          lineHeight: 1.1,
        }}
      >
        {pair.correlation != null ? pair.correlation.toFixed(2) : "—"}
      </div>

      <div style={{ marginBottom: "0.7rem", marginTop: "0.4rem" }}>
        <span
          style={{
            padding: "0.2rem 0.6rem",
            borderRadius: "20px",
            fontSize: "0.72rem",
            fontWeight: 600,
            background: labelBg,
            color: labelColor,
          }}
        >
          {pair.label}
        </span>
      </div>

      <GaugeBar corr={pair.correlation} label={pair.label} />

      <div
        style={{
          fontSize: "0.78rem",
          color: "var(--text-mute)",
          lineHeight: 1.5,
          marginTop: "0.7rem",
        }}
      >
        {pair.interpretation}
      </div>

      {pair.trend_sparkline && pair.trend_sparkline.length > 1 && (
        <div style={{ marginTop: "0.8rem" }}>
          <Sparkline data={pair.trend_sparkline} />
        </div>
      )}
    </div>
  );
}

export default function CorrelationMonitor({ pairs }) {
  if (!pairs || pairs.length === 0) return null;
  return (
    <>
      {pairs.map((pair) => (
        <CorrelationCard key={pair.key} pair={pair} />
      ))}
    </>
  );
}
