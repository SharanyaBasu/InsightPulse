export default function HeatmapGrid({ items, labelKey, valueKey, columns = 5 }) {
  function cellBg(value) {
    if (value == null) return "var(--panel)";
    const intensity = Math.min(Math.abs(value) / 6, 1);
    if (value > 0) return `rgba(0, 255, 65, ${0.06 + intensity * 0.28})`;
    if (value < 0) return `rgba(255, 51, 51, ${0.06 + intensity * 0.28})`;
    return "var(--panel)";
  }

  function valueColor(value) {
    if (value == null) return "var(--text-mute)";
    if (value > 0) return "var(--green)";
    if (value < 0) return "var(--red)";
    return "var(--text-soft)";
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: "3px",
      }}
    >
      {items.map((item, i) => (
        <div
          key={item[labelKey] || i}
          style={{
            background: cellBg(item[valueKey]),
            border: "1px solid var(--panel-border)",
            borderRadius: "var(--radius)",
            padding: "1rem 0.8rem",
            textAlign: "center",
          }}
        >
          <div
            style={{
              fontSize: "0.78rem",
              color: "var(--text-soft)",
              marginBottom: "0.35rem",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            {item[labelKey]}
          </div>
          <div
            style={{
              fontSize: "1.2rem",
              fontWeight: 700,
              color: valueColor(item[valueKey]),
            }}
          >
            {item[valueKey] != null
              ? `${item[valueKey] > 0 ? "+" : ""}${item[valueKey].toFixed(2)}%`
              : "—"}
          </div>
        </div>
      ))}
    </div>
  );
}
