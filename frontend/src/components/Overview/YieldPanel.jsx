export default function YieldPanel({ yieldData }) {
  const { ten_year, two_ten_slope_bps, slope_label } = yieldData;

  return (
    <div
      style={{
        background: "var(--panel)",
        padding: "1.5rem",
        borderRadius: "var(--radius)",
        border: "1px solid var(--panel-border)",
        margin: "2rem 0 1rem",
      }}
    >
      <h2 style={{ margin: 0, fontSize: "1.3rem", color: "var(--blue)" }}>
        Yield Curve
      </h2>

      <div style={{ marginTop: "1rem", fontSize: "1rem" }}>
        <div>10Y: {ten_year.toFixed(2)}%</div>
        <div>2s–10s: {two_ten_slope_bps.toFixed(1)} bps</div>
        <div style={{ color: "var(--text-soft)", marginTop: "0.4rem" }}>
          {slope_label}
        </div>
      </div>
    </div>
  );
}
