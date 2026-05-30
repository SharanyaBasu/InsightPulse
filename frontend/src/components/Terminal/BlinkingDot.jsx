export default function BlinkingDot({ color = "var(--green)", size = 6 }) {
  return (
    <span
      className="t-blink"
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRadius: "50%",
        background: color,
        flexShrink: 0,
      }}
    />
  );
}
