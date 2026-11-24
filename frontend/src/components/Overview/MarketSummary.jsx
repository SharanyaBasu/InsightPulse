import TerminalPanel from "../Terminal/TerminalPanel";

function bulletText(item) {
  return typeof item === "string" ? item : item?.text ?? "";
}

const sectionTitleStyle = (accent) => ({
  fontSize: "0.65rem",
  fontWeight: 600,
  color: accent,
  textTransform: "uppercase",
  letterSpacing: "0.1em",
  marginBottom: "0.35rem",
});

function BulletSection({ title, items, accent = "var(--cyan)", paragraph = false }) {
  if (paragraph) {
    if (Array.isArray(items)) {
      return (
        <BulletSection title={title} items={items} accent={accent} paragraph={false} />
      );
    }

    const text = bulletText(items);
    if (!text) return null;

    return (
      <div style={{ marginTop: "0.65rem" }}>
        <div style={sectionTitleStyle(accent)}>{title}</div>
        <p
          style={{
            color: "var(--text-soft)",
            fontSize: "0.78rem",
            lineHeight: 1.55,
            margin: 0,
          }}
        >
          {text}
        </p>
      </div>
    );
  }

  if (!items?.length) return null;

  return (
    <div style={{ marginTop: "0.65rem" }}>
        <div style={sectionTitleStyle(accent)}>{title}</div>
      <ul style={{ margin: 0, paddingLeft: "1.1rem", listStyle: "disc" }}>
        {items.map((item, index) => (
          <li
            key={`${title}-${index}`}
            style={{
              color: "var(--text-soft)",
              fontSize: "0.78rem",
              lineHeight: 1.55,
              marginBottom: "0.25rem",
            }}
          >
            {bulletText(item)}
          </li>
        ))}
      </ul>
    </div>
  );
}

function MoodStrip({ mood }) {
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
        <span style={{ color: "var(--text-mute)", flex: "1 1 100%" }}>
          {mood.drivers.map(bulletText).join(" · ")}
        </span>
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
  const validationTitle =
    validationStatus === "fallback"
      ? "LLM summary failed validation; showing structured data fallback."
      : validationStatus === "unverified"
        ? "Summary may not be accurate given current market data."
        : "LLM summary passed validation.";

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
              title={validationTitle}
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
          </div>

          {validationStatus === "unverified" && (
            <p
              style={{
                margin: "0 0 0.5rem 0",
                color: "var(--amber)",
                fontSize: "0.72rem",
                lineHeight: 1.45,
              }}
            >
              Summary may not be accurate given current market data.
            </p>
          )}

          <BulletSection
            title="Regime"
            items={summary.regime_summary}
            accent="var(--green)"
            paragraph
          />
          <details className="narrative-details">
            <summary>Read the briefing · key changes, risks & caveats</summary>
            <div className="briefing-grid">
              <BulletSection title="Key Changes" items={summary.key_changes} accent="var(--text-soft)" />
              <BulletSection title="Risks to Watch" items={summary.risks_to_watch} accent="var(--text-soft)" />
            </div>
            <MoodStrip mood={summary.mood_5d} />
            <BulletSection title="Caveats" items={summary.limitations} accent="var(--text-mute)" />
          </details>
        </>
      ) : (
        <ParagraphFallback text={paragraphText} />
      )}
    </TerminalPanel>
  );
}
