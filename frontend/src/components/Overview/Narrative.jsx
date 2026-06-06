export default function Narrative({ text }) {
  return (
    <div
      style={{
        background: "var(--panel)",
        padding: "0.5rem 0.8rem",
        border: "1px solid var(--panel-border)",
        borderRadius: "var(--radius)",
        marginBottom: "0.8rem",
        display: "flex",
        alignItems: "flex-start",
        gap: "0.5rem",
        fontSize: "0.8rem",
      }}
    >
      <span style={{ color: "var(--green)", fontWeight: 700, flexShrink: 0 }}>{">"}</span>
      <span style={{ color: "var(--text-soft)", lineHeight: 1.5 }}>{text}</span>
    </div>
  );
}
