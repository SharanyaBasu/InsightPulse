export default function TerminalLoader({ message = "LOADING DATA" }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        minHeight: "200px",
        color: "var(--green)",
        fontSize: "0.85rem",
        fontWeight: 600,
        letterSpacing: "0.08em",
      }}
    >
      {message}<span className="t-blink">_</span>
    </div>
  );
}
