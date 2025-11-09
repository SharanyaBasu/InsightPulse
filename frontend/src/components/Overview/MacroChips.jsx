export default function MacroChips({ macro }) {
  const entries = [
    ["CPI", macro.cpi.value, macro.cpi.direction],
    ["Unemployment", macro.unemployment.value, macro.unemployment.direction],
    ["Policy Rate", macro.policy_rate.value, macro.policy_rate.direction],
  ];

  const arrow = (d) =>
    d === "up" ? "↑" : d === "down" ? "↓" : "→";

  const arrowColor = (d) =>
    d === "up" ? "var(--red)" : d === "down" ? "var(--green)" : "var(--text-soft)";

  return (
    <div style={{ marginTop: "2rem" }}>
      <h3 style={{ color: "var(--blue)", marginBottom: "1rem" }}>
        Macro Snapshot
      </h3>

      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
        {entries.map(([label, value, dir]) => (
          <div
            key={label}
            style={{
              background: "var(--panel)",
              padding: "0.6rem 1rem",
              borderRadius: "var(--radius)",
              border: "1px solid var(--panel-border)",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
            }}
          >
            <strong>{label}:</strong>
            <span>{value.toFixed(2)}</span>
            <span style={{ color: arrowColor(dir), fontWeight: 700 }}>
              {arrow(dir)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
