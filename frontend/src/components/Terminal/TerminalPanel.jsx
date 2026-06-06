export default function TerminalPanel({ title, headerRight, children, style }) {
  return (
    <div
      style={{
        background: "var(--panel)",
        border: "1px solid var(--panel-border)",
        borderRadius: "var(--radius)",
        ...style,
      }}
    >
      {title && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0.4rem 0.8rem",
            borderBottom: "1px solid var(--panel-border)",
            fontSize: "0.7rem",
            fontWeight: 600,
            color: "var(--text-mute)",
            textTransform: "uppercase",
            letterSpacing: "0.12em",
          }}
        >
          <span>{title}</span>
          {headerRight}
        </div>
      )}
      <div style={{ padding: "0.6rem 0.8rem" }}>{children}</div>
    </div>
  );
}
