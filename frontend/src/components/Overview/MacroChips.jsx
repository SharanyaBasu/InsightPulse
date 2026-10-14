export default function MacroChips({ macro }) {
  const entries = [
    ["CPI", macro.cpi.value, macro.cpi.direction, macro.cpi.observed_on],
    ["UNEMP", macro.unemployment.value, macro.unemployment.direction, macro.unemployment.observed_on],
    ["RATE", macro.policy_rate.value, macro.policy_rate.direction, macro.policy_rate.observed_on],
  ];

  const arrow = (d) => (d === "up" ? "▲" : d === "down" ? "▼" : "—");
  const arrowColor = (d) => (d === "up" ? "var(--red)" : d === "down" ? "var(--green)" : "var(--text-mute)");

  return (
    <div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
        <tbody>
          {entries.map(([label, value, dir, observedOn]) => (
            <tr key={label} style={{ borderBottom: "1px solid var(--panel-border)" }}>
              <td style={{ padding: "0.5rem 0", color: "var(--text-mute)", width: "80px" }}>{label}<div style={{ fontSize: "0.65rem", marginTop: "0.2rem" }}>Observation: {observedOn || "unknown"}</div></td>
              <td style={{ padding: "0.5rem 0", textAlign: "right", fontWeight: 600, fontVariantNumeric: "tabular-nums", fontSize: "1rem" }}>
                {value == null ? "—" : value.toFixed(2)}
              </td>
              <td style={{ padding: "0.5rem 0", textAlign: "right", width: "35px", color: arrowColor(dir), fontWeight: 700, fontSize: "0.9rem" }}>
                {arrow(dir)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ fontSize: "0.7rem", color: "var(--text-mute)" }}>
        Observation dates identify the underlying periods, not publication or refresh times.
        {!entries.every((entry) => entry[3]) && " Original dates are unavailable for legacy forward-filled records; future ingestion preserves them."}
      </p>
    </div>
  );
}
