export default function YieldPanel({ yieldData }) {
  const { ten_year, two_ten_slope_bps, slope_label } = yieldData;

  const slopeColor = two_ten_slope_bps < 0 ? "var(--red)" : two_ten_slope_bps > 50 ? "var(--green)" : "var(--amber)";

  return (
    <div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
        <tbody>
          <tr style={{ borderBottom: "1px solid var(--panel-border)" }}>
            <td style={{ padding: "0.5rem 0", color: "var(--text-mute)" }}>10Y Yield</td>
            <td style={{ padding: "0.5rem 0", textAlign: "right", fontWeight: 600, fontVariantNumeric: "tabular-nums", fontSize: "1rem" }}>
              {ten_year.toFixed(2)}%
            </td>
          </tr>
          <tr style={{ borderBottom: "1px solid var(--panel-border)" }}>
            <td style={{ padding: "0.5rem 0", color: "var(--text-mute)" }}>2s-10s Slope</td>
            <td style={{ padding: "0.5rem 0", textAlign: "right", fontWeight: 600, color: slopeColor, fontVariantNumeric: "tabular-nums", fontSize: "1rem" }}>
              {two_ten_slope_bps.toFixed(1)} bps
            </td>
          </tr>
          <tr>
            <td style={{ padding: "0.5rem 0", color: "var(--text-mute)" }}>Curve</td>
            <td style={{ padding: "0.5rem 0", textAlign: "right", color: "var(--text-soft)", fontSize: "0.9rem" }}>
              {slope_label}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
