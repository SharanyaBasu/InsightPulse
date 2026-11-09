import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

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
  assetName: { fontSize: 12, color: "var(--text-soft)", fontWeight: 400, fontFamily: "'JetBrains Mono', monospace" },
  assetVal: { fontSize: 20, fontWeight: 500, lineHeight: 1.15 },
  assetUnit: { fontSize: 11, fontWeight: 400, fontFamily: "'JetBrains Mono', monospace" },
};

const POS_COLOR = "#1D9E75";
const NEG_COLOR = "#D85A30";

const OUTPUT_UNITS = {
  percent: "%",
  basis_points: "bps",
};

function formatValue(value, digits = 4) {
  if (!Number.isFinite(value)) return "—";
  const normalized = Object.is(value, -0) ? 0 : value;
  return normalized.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: digits,
  });
}

function ContributionTooltip({ active, payload, unit }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  const sign = row.contribution > 0 ? "+" : "";
  return (
    <div className="chart-tooltip">
      <strong>{row.label}</strong>
      <p>{row.submitted_value} {row.input_unit} submitted</p>
      <span>{sign}{formatValue(row.contribution)} {unit}</span>
    </div>
  );
}

function ContributionPanel({ asset, details }) {
  const unit = OUTPUT_UNITS[details.unit] || asset.unit;
  const rows = [...(details.contributions || [])]
    .filter((row) => row.contribution !== 0 || row.submitted_value !== 0)
    .sort((left, right) => Math.abs(right.contribution) - Math.abs(left.contribution));
  const maximum = Math.max(0, ...rows.map((row) => Math.abs(row.contribution)));
  const domainMaximum = maximum || 1;

  return (
    <section
      id="asset-contribution-panel"
      className="asset-contribution-panel"
      aria-labelledby="asset-contribution-heading"
    >
      <div className="asset-contribution-heading">
        <div>
          <p>Why this result?</p>
          <h3 id="asset-contribution-heading">{asset.label} model contributions</h3>
        </div>
        <span>{asset.value > 0 ? "+" : ""}{formatValue(asset.value, 2)} {asset.unit}</span>
      </div>

      <p className="asset-contribution-disclaimer">
        These are Ridge model contributions reflecting predictive associations, not proven causal effects.
      </p>

      {rows.length ? (
        <>
          <div className="asset-contribution-chart" style={{ height: Math.max(230, rows.length * 44) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={rows}
                layout="vertical"
                margin={{ top: 8, right: 18, bottom: 8, left: 8 }}
                accessibilityLayer
              >
                <CartesianGrid horizontal={false} stroke="var(--panel-border)" />
                <XAxis
                  type="number"
                  domain={[-domainMaximum, domainMaximum]}
                  tickFormatter={(value) => formatValue(value, 2)}
                  tick={{ fill: "var(--text-mute)", fontSize: 10 }}
                />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={118}
                  tick={{ fill: "var(--text-soft)", fontSize: 10 }}
                />
                <ReferenceLine x={0} stroke="var(--text-mute)" />
                <Tooltip content={<ContributionTooltip unit={unit} />} />
                <Bar dataKey="contribution" name="Model contribution" isAnimationActive={false}>
                  {rows.map((row) => (
                    <Cell
                      key={row.input_key}
                      fill={row.contribution >= 0 ? POS_COLOR : NEG_COLOR}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <dl className="asset-contribution-values" aria-label={`${asset.label} contribution values`}>
            {rows.map((row) => (
              <div key={row.input_key}>
                <dt>{row.label}</dt>
                <dd style={{ color: row.contribution >= 0 ? POS_COLOR : NEG_COLOR }}>
                  {row.contribution > 0 ? "+" : ""}{formatValue(row.contribution)} {unit}
                </dd>
              </div>
            ))}
          </dl>
        </>
      ) : (
        <p className="asset-contribution-empty">
          No scenario input contributed to this incremental result; all submitted shocks were neutral.
        </p>
      )}

      <div className="asset-contribution-reconciliation">
        <span>Contribution sum: {details.contribution_sum > 0 ? "+" : ""}{formatValue(details.contribution_sum)} {unit}</span>
        <span>Displayed total: {details.displayed_total > 0 ? "+" : ""}{formatValue(details.displayed_total, 2)} {unit}</span>
        <small>
          Full-precision contributions reconcile to the model result within {details.reconciliation_tolerance};
          the displayed total allows up to {details.display_rounding_tolerance} {unit} for rounding.
        </small>
      </div>
    </section>
  );
}

export default function ScenarioDeltaCards({ assets, contributions = {} }) {
  const [selectedKey, setSelectedKey] = useState(null);
  if (!assets?.length) return null;
  const selectedAsset = assets.find((asset) => asset.key === selectedKey);
  const selectedDetails = selectedKey ? contributions?.[selectedKey] : null;

  return (
    <div style={styles.root}>
      <p style={styles.sectionLabel}>Scenario delta — asset impact</p>
      <div style={styles.grid}>
        {assets.map((a) => {
          const isPos = a.value > 0;
          const sign = isPos ? "+" : "";
          const color = a.value === 0 ? "var(--text-soft)" : isPos ? POS_COLOR : NEG_COLOR;
          return (
            <div key={a.key || a.label} style={styles.card}>
              <span style={styles.assetName}>{a.label}</span>
              <span style={{ ...styles.assetVal, color }}>
                {sign}{a.value}
                <span style={styles.assetUnit}> {a.unit}</span>
              </span>
              <button
                type="button"
                className="asset-contribution-toggle"
                aria-expanded={selectedKey === a.key}
                aria-controls="asset-contribution-panel"
                disabled={!contributions?.[a.key]}
                onClick={() => setSelectedKey((current) => current === a.key ? null : a.key)}
              >
                Why this result?
              </button>
            </div>
          );
        })}
      </div>
      {selectedAsset && selectedDetails && (
        <ContributionPanel asset={selectedAsset} details={selectedDetails} />
      )}
    </div>
  );
}
