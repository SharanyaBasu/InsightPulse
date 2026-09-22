import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const SESSION_COUNT = 90;

const ASSET_META = {
  sp500: { label: "S&P 500", group: "Equities" },
  nasdaq: { label: "NASDAQ", group: "Equities" },
  irx: { label: "13-Week Treasury", group: "Rates" },
  fvx: { label: "5-Year Treasury", group: "Rates" },
  tnx: { label: "10-Year Treasury", group: "Rates" },
  usd_index: { label: "U.S. Dollar Index", group: "FX" },
  eurusd: { label: "EUR / USD", group: "FX" },
  gbpusd: { label: "GBP / USD", group: "FX" },
  audusd: { label: "AUD / USD", group: "FX" },
  usdjpy: { label: "USD / JPY", group: "FX" },
  usdchf: { label: "USD / CHF", group: "FX" },
  cew: { label: "Emerging Currency ETF", group: "FX" },
  gold: { label: "Gold", group: "Commodities" },
  oil: { label: "Crude Oil", group: "Commodities" },
  copper: { label: "Copper", group: "Commodities" },
  bitcoin: { label: "Bitcoin", group: "Crypto" },
  ethereum: { label: "Ethereum", group: "Crypto" },
};

const GROUP_ORDER = ["Equities", "Rates", "FX", "Commodities", "Crypto", "Other"];

function isFiniteNumber(value) {
  return value !== null && value !== "" && Number.isFinite(Number(value));
}

function pearsonCorrelation(valuesA, valuesB) {
  if (valuesA.length !== valuesB.length || valuesA.length < 2) return null;

  const meanA = valuesA.reduce((sum, value) => sum + value, 0) / valuesA.length;
  const meanB = valuesB.reduce((sum, value) => sum + value, 0) / valuesB.length;

  let numerator = 0;
  let varianceA = 0;
  let varianceB = 0;

  for (let index = 0; index < valuesA.length; index += 1) {
    const deltaA = valuesA[index] - meanA;
    const deltaB = valuesB[index] - meanB;
    numerator += deltaA * deltaB;
    varianceA += deltaA ** 2;
    varianceB += deltaB ** 2;
  }

  const denominator = Math.sqrt(varianceA * varianceB);
  return denominator === 0 ? null : numerator / denominator;
}

function correlationLabel(value) {
  if (value == null) return "Insufficient data";
  if (value >= 0.7) return "Strong positive";
  if (value >= 0.3) return "Positive";
  if (value <= -0.7) return "Strong negative";
  if (value <= -0.3) return "Negative";
  return "Weak relationship";
}

function correlationTone(value) {
  if (value == null) return "neutral";
  if (value >= 0.3) return "positive";
  if (value <= -0.3) return "negative";
  return "neutral";
}

function formatRawValue(value) {
  if (!isFiniteNumber(value)) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatDate(value) {
  const date = new Date(`${String(value).slice(0, 10)}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function ComparisonTooltip({ active, payload, label, assetA, assetB }) {
  if (!active || !payload?.length) return null;

  const point = payload[0]?.payload;
  const rows = [
    { key: "assetA", meta: assetA, color: "var(--green)" },
    { key: "assetB", meta: assetB, color: "var(--cyan)" },
  ];

  return (
    <div className="correlation-tooltip">
      <div className="correlation-tooltip-date">{formatDate(label)}</div>
      {rows.map(({ key, meta, color }) => (
        <div className="correlation-tooltip-row" key={key}>
          <span className="correlation-tooltip-name" style={{ color }}>{meta.label}</span>
          <strong>{point?.[`${key}Indexed`]?.toFixed(2) ?? "—"}</strong>
          <span>{formatRawValue(point?.[`${key}Raw`])} raw</span>
        </div>
      ))}
    </div>
  );
}

function AssetSelect({ id, label, value, onChange, assets, disabledValue }) {
  const groupedAssets = GROUP_ORDER.map((group) => ({
    group,
    assets: assets.filter((asset) => asset.group === group),
  })).filter(({ assets: groupAssets }) => groupAssets.length > 0);

  return (
    <label className="correlation-select" htmlFor={id}>
      <span>{label}</span>
      <select id={id} value={value} onChange={(event) => onChange(event.target.value)}>
        {groupedAssets.map(({ group, assets: groupAssets }) => (
          <optgroup key={group} label={group}>
            {groupAssets.map((asset) => (
              <option key={asset.key} value={asset.key} disabled={asset.key === disabledValue}>
                {asset.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}

export default function CrossAssetCorrelation({ history }) {
  const [selectedA, setSelectedA] = useState("sp500");
  const [selectedB, setSelectedB] = useState("gold");

  const assets = useMemo(() => {
    if (!Array.isArray(history)) return [];

    const keys = new Set();
    history.forEach((row) => {
      Object.keys(row || {}).forEach((key) => {
        if (key !== "date" && key !== "id" && isFiniteNumber(row[key])) keys.add(key);
      });
    });

    return [...keys]
      .map((key) => ({
        key,
        label: ASSET_META[key]?.label || key.replaceAll("_", " ").toUpperCase(),
        group: ASSET_META[key]?.group || "Other",
      }))
      .sort((left, right) => {
        const groupDelta = GROUP_ORDER.indexOf(left.group) - GROUP_ORDER.indexOf(right.group);
        return groupDelta || left.label.localeCompare(right.label);
      });
  }, [history]);

  const assetAKey = assets.some((asset) => asset.key === selectedA) ? selectedA : assets[0]?.key;
  const fallbackB = assets.find((asset) => asset.key === "gold" && asset.key !== assetAKey)
    || assets.find((asset) => asset.key !== assetAKey);
  const assetBKey = assets.some((asset) => asset.key === selectedB && asset.key !== assetAKey)
    ? selectedB
    : fallbackB?.key;

  const assetA = assets.find((asset) => asset.key === assetAKey);
  const assetB = assets.find((asset) => asset.key === assetBKey);

  const comparison = useMemo(() => {
    if (!assetAKey || !assetBKey || !Array.isArray(history)) return null;

    const overlapping = history
      .filter((row) => isFiniteNumber(row?.[assetAKey]) && isFiniteNumber(row?.[assetBKey]))
      .slice(-SESSION_COUNT)
      .map((row) => ({
        date: row.date,
        assetARaw: Number(row[assetAKey]),
        assetBRaw: Number(row[assetBKey]),
      }));

    if (overlapping.length < 3) return null;

    const firstA = overlapping[0].assetARaw;
    const firstB = overlapping[0].assetBRaw;
    const data = overlapping.map((row) => ({
      ...row,
      assetAIndexed: (row.assetARaw / firstA) * 100,
      assetBIndexed: (row.assetBRaw / firstB) * 100,
    }));

    const returnsA = [];
    const returnsB = [];
    for (let index = 1; index < overlapping.length; index += 1) {
      const previous = overlapping[index - 1];
      const current = overlapping[index];
      returnsA.push(current.assetARaw / previous.assetARaw - 1);
      returnsB.push(current.assetBRaw / previous.assetBRaw - 1);
    }

    return {
      data,
      correlation: pearsonCorrelation(returnsA, returnsB),
      startDate: overlapping[0].date,
      endDate: overlapping.at(-1).date,
      changeA: data.at(-1).assetAIndexed - 100,
      changeB: data.at(-1).assetBIndexed - 100,
      sessionCount: overlapping.length,
    };
  }, [assetAKey, assetBKey, history]);

  if (assets.length < 2) {
    return (
      <div className="cross-asset-state" role="status">
        <strong>NOT ENOUGH COMPARABLE ASSETS</strong>
        <span>At least two historical price series are required.</span>
      </div>
    );
  }

  if (!comparison || !assetA || !assetB) {
    return (
      <div className="cross-asset-state" role="status">
        <strong>NO OVERLAPPING HISTORY</strong>
        <span>Choose another asset pair to calculate a correlation.</span>
      </div>
    );
  }

  const tone = correlationTone(comparison.correlation);
  const dateRange = `${formatDate(comparison.startDate)} — ${formatDate(comparison.endDate)}`;
  const chartLabel = `${assetA.label} and ${assetB.label} normalized performance from ${dateRange}`;

  return (
    <div className="cross-asset-page">
      <header className="cross-asset-header">
        <div>
          <p className="cross-asset-kicker">RELATIONSHIP ANALYSIS</p>
          <h1>Cross-Asset Correlation</h1>
          <p>Compare two markets on a common baseline and measure how their daily returns move together.</p>
        </div>
        <div className="correlation-window" aria-label="Analysis window">
          <span>{comparison.sessionCount}</span>
          trading sessions
        </div>
      </header>

      <section className="correlation-workspace" aria-labelledby="correlation-chart-title">
        <div className="correlation-toolbar">
          <div className="correlation-pair-controls">
            <AssetSelect
              id="correlation-asset-a"
              label="Primary asset"
              value={assetAKey}
              onChange={setSelectedA}
              assets={assets}
              disabledValue={assetBKey}
            />
            <span className="correlation-versus" aria-hidden="true">VS</span>
            <AssetSelect
              id="correlation-asset-b"
              label="Comparison asset"
              value={assetBKey}
              onChange={setSelectedB}
              assets={assets}
              disabledValue={assetAKey}
            />
          </div>

          <div className={`correlation-result is-${tone}`} aria-live="polite">
            <span>Return correlation</span>
            <strong>{comparison.correlation?.toFixed(2) ?? "—"}</strong>
            <small>{correlationLabel(comparison.correlation)}</small>
          </div>
        </div>

        <div className="correlation-chart-header">
          <div>
            <h2 id="correlation-chart-title">Normalized performance</h2>
            <p>Both series begin at 100 so relative direction is directly comparable.</p>
          </div>
          <span>{dateRange}</span>
        </div>

        <div className="correlation-chart" role="img" aria-label={chartLabel}>
          <ResponsiveContainer
            width="100%"
            height="100%"
            initialDimension={{ width: 800, height: 400 }}
          >
            <LineChart data={comparison.data} margin={{ top: 12, right: 18, bottom: 4, left: 4 }}>
              <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.055)" />
              <XAxis
                dataKey="date"
                tickFormatter={(value) => formatDate(value).replace(/, \d{4}/, "")}
                tick={{ fontSize: 10, fill: "#787878", fontFamily: "inherit" }}
                tickLine={false}
                axisLine={{ stroke: "var(--panel-border)" }}
                minTickGap={48}
              />
              <YAxis
                domain={["auto", "auto"]}
                tickFormatter={(value) => value.toFixed(0)}
                tick={{ fontSize: 10, fill: "#787878", fontFamily: "inherit" }}
                tickLine={false}
                axisLine={false}
                width={38}
              />
              <ReferenceLine y={100} stroke="var(--text-mute)" strokeDasharray="4 5" />
              <Tooltip
                content={<ComparisonTooltip assetA={assetA} assetB={assetB} />}
                cursor={{ stroke: "var(--text-soft)", strokeDasharray: "3 4" }}
              />
              <Line
                type="monotone"
                dataKey="assetAIndexed"
                name={assetA.label}
                stroke="var(--green)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: "var(--green)", stroke: "var(--bg)", strokeWidth: 2 }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="assetBIndexed"
                name={assetB.label}
                stroke="var(--cyan)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: "var(--cyan)", stroke: "var(--bg)", strokeWidth: 2 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="correlation-legend" aria-label="Series performance summary">
          <div>
            <span className="correlation-swatch is-primary" />
            <strong>{assetA.label}</strong>
            <span className={comparison.changeA >= 0 ? "is-up" : "is-down"}>
              {comparison.changeA >= 0 ? "+" : ""}{comparison.changeA.toFixed(2)}%
            </span>
          </div>
          <div>
            <span className="correlation-swatch is-comparison" />
            <strong>{assetB.label}</strong>
            <span className={comparison.changeB >= 0 ? "is-up" : "is-down"}>
              {comparison.changeB >= 0 ? "+" : ""}{comparison.changeB.toFixed(2)}%
            </span>
          </div>
          <p>Correlation uses daily percentage returns across the displayed period.</p>
        </div>
      </section>
    </div>
  );
}
