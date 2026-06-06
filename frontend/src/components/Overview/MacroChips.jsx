export default function MacroChips({ macro }) {
  const entries = [
    ["CPI", macro.cpi.value, macro.cpi.direction],
    ["UNEMP", macro.unemployment.value, macro.unemployment.direction],
    ["RATE", macro.policy_rate.value, macro.policy_rate.direction],
  ];

  const arrow = (d) => (d === "up" ? "▲" : d === "down" ? "▼" : "—");
  const arrowColor = (d) => (d === "up" ? "var(--red)" : d === "down" ? "var(--green)" : "var(--text-mute)");

  return (
    <div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
        <tbody>
          {entries.map(([label, value, dir]) => (
            <tr key={label} style={{ borderBottom: "1px solid var(--panel-border)" }}>
              <td style={{ padding: "0.5rem 0", color: "var(--text-mute)", width: "80px" }}>{label}</td>
              <td style={{ padding: "0.5rem 0", textAlign: "right", fontWeight: 600, fontVariantNumeric: "tabular-nums", fontSize: "1rem" }}>
                {value.toFixed(2)}
              </td>
              <td style={{ padding: "0.5rem 0", textAlign: "right", width: "35px", color: arrowColor(dir), fontWeight: 700, fontSize: "0.9rem" }}>
                {arrow(dir)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
