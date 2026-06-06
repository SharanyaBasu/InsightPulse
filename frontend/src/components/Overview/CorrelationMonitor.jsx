import Sparkline from "./Sparkline";

function corrColor(corr) {
  if (corr == null) return "var(--text-mute)";
  if (corr >= 0.3) return "var(--green)";
  if (corr <= -0.3) return "var(--red)";
  return "var(--text-soft)";
}

function labelColor(label) {
  if (label === "Positive") return "var(--green)";
  if (label === "Negative") return "var(--red)";
  return "var(--text-soft)";
}

function GaugeBar({ corr, label }) {
  if (corr == null) return null;
  const color = labelColor(label);
  // Build ASCII-style bar
  const filled = Math.round(Math.abs(corr) * 8);
  const empty = 8 - filled;
  const bar = (corr >= 0 ? "" : "") + "\u2588".repeat(filled) + "\u2500".repeat(empty);

  return (
    <div style={{ fontFamily: "inherit", fontSize: "0.7rem", color, letterSpacing: "0.05em", margin: "0.3rem 0" }}>
      [{bar}] {corr.toFixed(2)}
    </div>
  );
}

function CorrelationCard({ pair }) {
  return (
    <div style={{ background: "var(--panel)", border: "1px solid var(--panel-border)", borderRadius: "var(--radius)", padding: "0.6rem 0.8rem" }}>
      <div style={{ fontSize: "0.65rem", color: "var(--text-mute)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "0.3rem" }}>
        {pair.pair}
      </div>

      <div style={{ fontSize: "1.4rem", fontWeight: 700, color: corrColor(pair.correlation), lineHeight: 1.1 }}>
        {pair.correlation != null ? pair.correlation.toFixed(2) : "—"}
      </div>

      <span style={{ fontSize: "0.6rem", fontWeight: 600, color: labelColor(pair.label), textTransform: "uppercase" }}>
        [{pair.label}]
      </span>

      <GaugeBar corr={pair.correlation} label={pair.label} />

      <div style={{ fontSize: "0.7rem", color: "var(--text-mute)", lineHeight: 1.4 }}>
        {pair.interpretation}
      </div>

      {pair.trend_sparkline && pair.trend_sparkline.length > 1 && (
        <div style={{ marginTop: "0.4rem" }}>
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
