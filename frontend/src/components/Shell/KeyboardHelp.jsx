export default function KeyboardHelp({ onClose }) {
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.75)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 100,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--panel)",
          border: "1px solid var(--panel-border)",
          padding: "1.2rem 1.6rem",
          minWidth: 320,
          fontSize: "0.8rem",
        }}
      >
        <div style={{ color: "var(--green)", fontWeight: 700, marginBottom: "1rem", fontSize: "0.85rem" }}>
          KEYBOARD SHORTCUTS
        </div>
        {[
          ["1-5", "Navigate pages"],
          ["r", "Refresh data"],
          ["?", "Toggle this help"],
          ["Esc", "Close overlay"],
        ].map(([key, desc]) => (
          <div
            key={key}
            style={{
              display: "flex",
              justifyContent: "space-between",
              padding: "0.3rem 0",
              borderBottom: "1px solid var(--panel-border)",
            }}
          >
            <span className="t-kbd" style={{ minWidth: 40, textAlign: "center" }}>{key}</span>
            <span style={{ color: "var(--text-soft)" }}>{desc}</span>
          </div>
        ))}
        <div style={{ color: "var(--text-mute)", marginTop: "0.8rem", fontSize: "0.7rem" }}>
          Press Esc or click outside to close
        </div>
      </div>
    </div>
  );
}
