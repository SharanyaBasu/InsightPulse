export default function Narrative({ text }) {
  return (
    <div
      style={{
        background: "var(--panel)",
        padding: "1rem 1.4rem",
        borderRadius: "var(--radius)",
        border: "1px solid var(--panel-border)",
        boxShadow: "0 0 10px var(--panel-glow)",
        marginBottom: "1.6rem",
        display: "flex",
        alignItems: "center",
        gap: "0.8rem",
      }}
    >
      <div
        style={{
          width: "8px",
          height: "8px",
          borderRadius: "50%",
          background: "var(--blue)",
        }}
      />
      <span style={{ fontSize: "1.05rem", color: "var(--text-soft)" }}>
        {text}
      </span>
    </div>
  );
}
