const styles = {
  root: { fontFamily: "sans-serif", padding: "1.25rem 0" },
  sectionLabel: {
    fontSize: 11,
    fontWeight: 500,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "var(--text-soft)",
    margin: "0 0 10px",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
    gap: 10,
  },
  card: {
    background: "rgba(28, 33, 40, 0.85)",
    border: "1px solid var(--panel-border)",
    borderRadius: "var(--radius)",
    padding: "14px 14px 12px",
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  assetName: { fontSize: 12, color: "var(--text-soft)", fontWeight: 400 },
  assetVal: { fontSize: 20, fontWeight: 500, lineHeight: 1.15 },
  assetUnit: { fontSize: 11, fontWeight: 400 },
};

const POS_COLOR = "#1D9E75";
const NEG_COLOR = "#D85A30";

export default function ScenarioDeltaCards({ assets }) {
  if (!assets?.length) return null;
  return (
    <div style={styles.root}>
      <p style={styles.sectionLabel}>Scenario delta — asset impact</p>
      <div style={styles.grid}>
        {assets.map((a) => {
          const isPos = a.value >= 0;
          const sign = isPos ? "+" : "";
          const color = isPos ? POS_COLOR : NEG_COLOR;
          return (
            <div key={a.label} style={styles.card}>
              <span style={styles.assetName}>{a.label}</span>
              <span style={{ ...styles.assetVal, color }}>
                {sign}{a.value}
                <span style={styles.assetUnit}> {a.unit}</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
