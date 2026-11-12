import TerminalPanel from "../Terminal/TerminalPanel";

function bulletText(item) {
  return typeof item === "string" ? item : item?.text ?? "";
}

function evidenceIds(item) {
  if (!item || typeof item !== "object" || !Array.isArray(item.evidence_ids)) return [];
  return item.evidence_ids.filter((id) => typeof id === "string" && id);
}

const ASSET_LABELS = {
  sp500: "S&P 500",
  nasdaq: "NASDAQ",
  usd_index: "Dollar index",
  dxy: "Dollar index",
  vix: "VIX",
  tlt: "Treasury bonds",
  gold: "Gold",
  bitcoin: "Bitcoin",
  ethereum: "Ethereum",
};

const MEASURE_LABELS = [
  ["5d_return", "5-day return"],
  ["1d_return", "1-day return"],
  ["20d_volatility", "20-day volatility"],
  ["drawdown_from_peak", "drawdown from peak"],
  ["price_zscore_252d", "price z-score"],
  ["ma_distance_50d", "distance from the 50-day average"],
  ["ma_distance_200d", "distance from the 200-day average"],
];

function indicatorLabel(id) {
  if (id === "yield_curve_slope_10y_2y") return "10-year minus 2-year yield slope";
  if (id === "real_rate_proxy") return "Real rate proxy";
  if (id === "btc_eth_ratio") return "Bitcoin/Ethereum ratio";
  for (const [asset, label] of Object.entries(ASSET_LABELS)) {
    if (id === asset || id.startsWith(`${asset}_`)) {
      const measure = MEASURE_LABELS.find(([key]) => id.slice(asset.length + 1) === key);
      return measure ? `${label}, ${measure[1]}` : label;
    }
  }
  return id.replaceAll("_", " ");
}

function formatMeasurement(row) {
  if (row.display_value == null || row.display_value === "") return "Value not recorded";
  const numeric = typeof row.display_value === "number" ? row.display_value : Number(row.display_value);
  const shown = Number.isFinite(numeric)
    ? numeric.toLocaleString("en-US", { maximumFractionDigits: 3 })
    : String(row.display_value);
  if (row.unit === "pct") return `${shown}%`;
  if (row.unit === "pct_annualized") return `${shown}% annualized`;
  if (row.unit && row.unit !== "level") return `${shown} ${row.unit}`;
  return shown;
}

function formatPeriod(window) {
  if (!window) return "Period not recorded";
  const days = String(window).match(/^(\d+)d$/);
  if (days) return `${days[1]}-day window`;
  if (window === "1y") return "1-year window";
  if (window === "available_history") return "Available history";
  return String(window);
}

function formatObserved(timestamp) {
  if (!timestamp) return "Date not recorded";
  const match = String(timestamp).match(/\d{4}-\d{2}-\d{2}/);
  return match ? match[0] : String(timestamp);
}

function formatSource(row) {
  const dependencies = Array.isArray(row.source_dependencies)
    ? row.source_dependencies.filter((item) => typeof item === "string" && item)
    : [];
  if (dependencies.length) return dependencies.join(", ");
  if (typeof row.source === "string" && row.source) return row.source.replaceAll("_", " ");
  return "calculated metrics";
}

function EvidenceList({ ids, evidence }) {
  const resolved = ids
    .map((id) => [id, evidence?.[id]])
    .filter(([, row]) => row && typeof row === "object");
  if (!resolved.length) {
    return <p className="claim-note">No supporting evidence</p>;
  }
  return (
    <ul className="claim-evidence">
      {resolved.map(([id, row]) => (
        <li key={id}>
          <strong>{indicatorLabel(id)}</strong>
          <span>{formatMeasurement(row)}</span>
          <span>{formatPeriod(row.window)}</span>
          <span>Observed {formatObserved(row.timestamp)}</span>
          <span>Source: {formatSource(row)}</span>
          <span className="claim-id">{id}</span>
        </li>
      ))}
    </ul>
  );
}

function Claim({ item, evidence }) {
  const text = bulletText(item);
  if (!text) return null;
  return (
    <details className="claim">
      <summary>{text}</summary>
      <EvidenceList ids={evidenceIds(item)} evidence={evidence} />
    </details>
  );
}

const sectionTitleStyle = (accent) => ({
  fontSize: "0.65rem",
  fontWeight: 600,
  color: accent,
  textTransform: "uppercase",
  letterSpacing: "0.1em",
  marginBottom: "0.35rem",
});

function BulletSection({ title, items, accent = "var(--cyan)", evidence = {}, sourced = true }) {
  const claims = (Array.isArray(items) ? items : items ? [items] : [])
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => bulletText(item));

  if (!claims.length) return null;

  return (
    <div style={{ marginTop: "0.65rem" }}>
      <div style={sectionTitleStyle(accent)}>{title}</div>
      {claims.map(({ item, index }) => (
        sourced ? (
          <Claim key={`${title}-${index}`} item={item} evidence={evidence} />
        ) : (
          <p key={`${title}-${index}`} className="claim-note" style={{ color: "var(--text-soft)" }}>
            {bulletText(item)}
          </p>
        )
      ))}
    </div>
  );
}

function MoodStrip({ mood, evidence }) {
  if (!mood) return null;

  const label = mood.label || "neutral";
  const color =
    label === "bullish"
      ? "var(--green)"
      : label === "bearish"
        ? "var(--red)"
        : "var(--amber)";
  const prob =
    mood.probability != null
      ? `${Math.round(mood.probability * 100)}%`
      : null;

  return (
    <div
      style={{
        marginTop: "0.65rem",
        padding: "0.45rem 0.55rem",
        border: "1px solid var(--panel-border)",
        borderRadius: "var(--radius)",
        display: "flex",
        flexWrap: "wrap",
        gap: "0.5rem 1rem",
        alignItems: "baseline",
        fontSize: "0.75rem",
      }}
    >
      <span style={{ color: "var(--text-mute)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        5D Mood
      </span>
      <span style={{ color, fontWeight: 700 }}>{label}</span>
      {prob && (
        <span style={{ color: "var(--text-soft)" }}>confidence {prob}</span>
      )}
      {mood.drivers?.length > 0 && (
        <div style={{ flex: "1 1 100%" }}>
          {mood.drivers.map((driver, index) => (
            <Claim key={`mood-${index}`} item={driver} evidence={evidence} />
          ))}
        </div>
      )}
    </div>
  );
}

function ParagraphFallback({ text }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "0.5rem",
        fontSize: "0.82rem",
        lineHeight: 1.6,
      }}
    >
      <span style={{ color: "var(--green)", fontWeight: 700, flexShrink: 0 }}>
        {">"}
      </span>
      <p style={{ margin: 0, color: "var(--text-soft)" }}>{text}</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div role="status" style={{ color: "var(--text-mute)", fontSize: "0.8rem", lineHeight: 1.6 }}>
      <span style={{ color: "var(--amber)" }}>{">"}</span> Generating market summary…
    </div>
  );
}

function FallbackState({ message, onRetry }) {
  return (
    <div style={{ fontSize: "0.8rem", lineHeight: 1.6 }}>
      <p style={{ margin: "0 0 0.5rem 0", color: "var(--text-soft)" }}>
        <span style={{ color: "var(--amber)", fontWeight: 700 }}>{">"}</span> {message}
      </p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          style={{
            background: "transparent",
            border: "1px solid var(--panel-border)",
            color: "var(--cyan)",
            padding: "0.25rem 0.5rem",
            fontSize: "0.7rem",
            cursor: "pointer",
            borderRadius: "var(--radius)",
            fontFamily: "inherit",
          }}
        >
          Retry
        </button>
      )}
    </div>
  );
}

const REGIME_LABELS = {
  risk_on: "Risk-On",
  risk_off: "Risk-Off",
  transitional: "Neutral",
  inflationary: "Inflationary",
  liquidity_stress: "Liquidity Stress",
};

export default function MarketSummary({ summary, loading, error, onRetry }) {
  if (loading && !summary) {
    return (
      <TerminalPanel title="Market takeaway" className="takeaway-panel" style={{ marginBottom: "0.5rem" }}>
        <LoadingState />
      </TerminalPanel>
    );
  }

  if (error && !summary) {
    return (
      <TerminalPanel title="Market takeaway" className="takeaway-panel" style={{ marginBottom: "0.5rem" }}>
        <FallbackState
          message={error}
          onRetry={onRetry}
        />
      </TerminalPanel>
    );
  }

  const isStructured = Boolean(summary?.headline);
  const paragraphText =
    typeof summary?.summary === "string" && summary.summary.trim()
      ? summary.summary.trim()
      : null;

  if (!isStructured && !paragraphText) {
    return (
      <TerminalPanel title="Market takeaway" className="takeaway-panel" style={{ marginBottom: "0.5rem" }}>
        <FallbackState
          message="No daily summary available yet."
          onRetry={onRetry}
        />
      </TerminalPanel>
    );
  }

  const regimeLabel = summary.regime_label
    ? REGIME_LABELS[summary.regime_label] || summary.regime_label
    : null;
  const validationStatus = summary.validation?.status;
  const validationLabel =
    validationStatus === "verified"
      ? "VERIFIED"
      : validationStatus === "unverified"
        ? "MAY BE INACCURATE"
        : validationStatus === "fallback"
          ? "DATA FALLBACK"
          : null;
  const validationColor =
    validationStatus === "verified"
      ? "var(--green)"
      : "var(--amber)";
  const validationCopy =
    validationStatus === "verified"
      ? "Checked against today's recorded indicators. Not a prediction or a guarantee."
      : validationStatus === "unverified"
        ? "This wording was not confirmed against today's recorded indicators. Not a prediction or a guarantee."
        : validationStatus === "fallback"
          ? "Showing recorded indicators because the written summary failed its check. Not a prediction or a guarantee."
          : null;
  const evidence = summary.evidence && typeof summary.evidence === "object" ? summary.evidence : {};

  return (
    <TerminalPanel
      title="Market takeaway" className="takeaway-panel"
      style={{ marginBottom: "0.5rem" }}
      headerRight={
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
          {regimeLabel && (
            <span
              style={{
                fontSize: "0.65rem",
                color: "var(--text-mute)",
                letterSpacing: "0.06em",
              }}
            >
              {regimeLabel.toUpperCase()}
            </span>
          )}
          {validationLabel && (
            <span
              style={{
                fontSize: "0.65rem",
                color: validationColor,
                letterSpacing: "0.06em",
              }}
            >
              {validationLabel}
            </span>
          )}
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--text-mute)",
                fontSize: "0.65rem",
                cursor: "pointer",
                fontFamily: "inherit",
                letterSpacing: "0.06em",
              }}
            >
              REFRESH
            </button>
          )}
        </div>
      }
    >
      {summary && (loading || error) && (
        <p role="status" style={{ color: "var(--amber)", fontSize: "0.75rem" }}>
          {loading ? "Checking for an updated summary…" : `${error} Showing the previous summary.`}
        </p>
      )}
      {summary?.data_as_of && (
        <p style={{ color: "var(--text-mute)", fontSize: "0.72rem" }}>
          Underlying observations: {summary.data_as_of.oldest} – {summary.data_as_of.latest}
        </p>
      )}
      {isStructured ? (
        <>
          <div
            style={{
              borderLeft: "2px solid var(--green)",
              paddingLeft: "0.65rem",
              marginBottom: "0.5rem",
            }}
          >
            <div
              style={{
                color: "var(--text)",
                fontSize: "clamp(1.2rem, 2vw, 1.7rem)",
                fontWeight: 700,
                lineHeight: 1.45,
              }}
            >
              {summary.headline}
            </div>
            <p className="claim-note">Not individually sourced. The claims below carry the supporting indicators.</p>
          </div>

          {validationCopy && (
            <p className="claim-note" style={{ color: validationColor }}>
              {validationCopy}
            </p>
          )}

          <BulletSection
            title="Regime"
            items={summary.regime_summary}
            accent="var(--green)"
            evidence={evidence}
          />
          <details className="narrative-details">
            <summary>Read the briefing · key changes, risks & caveats</summary>
            <div className="briefing-grid">
              <BulletSection title="Key Changes" items={summary.key_changes} accent="var(--text-soft)" evidence={evidence} />
              <BulletSection title="Risks to Watch" items={summary.risks_to_watch} accent="var(--text-soft)" evidence={evidence} />
            </div>
            <MoodStrip mood={summary.mood_5d} evidence={evidence} />
            <BulletSection title="Caveats" items={summary.limitations} accent="var(--text-mute)" sourced={false} />
          </details>
        </>
      ) : (
        <ParagraphFallback text={paragraphText} />
      )}
    </TerminalPanel>
  );
}
