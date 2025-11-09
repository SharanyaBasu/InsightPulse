export default function SectorTile({ tile }) {
  const c = tile.change_1m;

  const color =
    c > 0 ? "var(--green)" : c < 0 ? "var(--red)" : "var(--text-soft)";

  return (
    <div
      style={{
        background: "var(--panel)",
        padding: "1.2rem",
        borderRadius: "var(--radius)",
        border: "1px solid var(--panel-border)",
        boxShadow: "0 0 10px var(--panel-glow)",
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: "1.05rem", fontWeight: 600 }}>{tile.sector}</div>
      <div style={{ marginTop: "0.4rem", fontSize: "0.95rem", color }}>
        {c.toFixed(2)}%
      </div>
    </div>
  );
}
