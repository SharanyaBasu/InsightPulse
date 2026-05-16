export default function RegionTile({ tile }) {
  const color = tile.change_1m > 0 ? "var(--green)" : tile.change_1m < 0 ? "var(--red)" : "var(--text-soft)";

  return (
    <div
      style={{
        background: "var(--panel)",
        border: "1px solid var(--panel-border)",
        borderRadius: "var(--radius)",
        padding: "0.5rem 0.6rem",
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {tile.region}
      </div>
      <div style={{ fontSize: "0.95rem", fontWeight: 700, color, marginTop: "0.2rem", fontVariantNumeric: "tabular-nums" }}>
        {tile.change_1m > 0 ? "+" : ""}{tile.change_1m.toFixed(2)}%
      </div>
    </div>
  );
}
