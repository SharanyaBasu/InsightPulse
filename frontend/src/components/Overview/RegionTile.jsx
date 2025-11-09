export default function RegionTile({ tile }) {
  const color =
    tile.change_1m > 0
      ? "var(--green)"
      : tile.change_1m < 0
      ? "var(--red)"
      : "var(--text-soft)";

  return (
    <div
      style={{
        background: "var(--panel)",
        padding: "1rem",
        borderRadius: "var(--radius)",
        border: "1px solid var(--panel-border)",
        textAlign: "center",
        boxShadow: "0 0 10px var(--panel-glow)",
      }}
    >
      <div style={{ fontSize: "1rem", fontWeight: 600 }}>{tile.region}</div>
      <div style={{ marginTop: "0.4rem", color }}>{tile.change_1m.toFixed(2)}%</div>
    </div>
  );
}
