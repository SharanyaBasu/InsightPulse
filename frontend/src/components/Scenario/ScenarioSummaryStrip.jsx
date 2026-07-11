export default function ScenarioSummaryStrip({ summary }) {
  if (!summary) return null;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.6rem",
        padding: "0.6rem 0.8rem",
        background: "rgba(0, 255, 65, 0.04)",
        border: "1px solid var(--panel-border)",
        borderLeft: "3px solid var(--green)",
        borderRadius: "var(--radius)",
        marginBottom: "0.6rem",
      }}
    >
      <span
        style={{
          fontSize: "0.62rem",
          fontWeight: 700,
          color: "var(--green)",
          textTransform: "uppercase",
          letterSpacing: "0.14em",
          flexShrink: 0,
        }}
      >
        Summary
      </span>
      <span
        style={{
          fontSize: "0.92rem",
          color: "var(--text)",
          lineHeight: 1.45,
        }}
      >
        {summary}
      </span>
    </div>
  );
}
