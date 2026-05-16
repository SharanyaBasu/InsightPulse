const REGIME_ACCENT = {
  "Growth Regime": "var(--green)",
  "Inflationary Regime": "var(--amber)",
  "Liquidity Stress": "var(--red)",
  "Transitional / Unstable": "var(--cyan)",
};

function SignalRow({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "0.2rem 0", fontSize: "0.78rem" }}>
      <span style={{ color: "var(--text-mute)" }}>{label}</span>
      <span style={{ color: "var(--text)", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>{value ?? "—"}</span>
    </div>
  );
}

function SignalBlock({ title, children }) {
  return (
    <div>
      <div style={{ fontSize: "0.65rem", color: "var(--text-mute)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "0.35rem", fontWeight: 600 }}>
        {title}
      </div>
      {children}
    </div>
  );
}

export default function RegimeLabel({ regime }) {
  if (!regime) return null;

  const accent = REGIME_ACCENT[regime.regime] || "var(--cyan)";
  const { vix, dispersion, correlations } = regime.signals || {};

  return (
    <div style={{ background: "var(--panel)", border: "1px solid var(--panel-border)", borderRadius: "var(--radius)", padding: "0.8rem" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.4rem", flexWrap: "wrap" }}>
        <span style={{ color: accent, fontSize: "0.85rem", fontWeight: 700 }}>
          {">"} {regime.regime}
        </span>
        <span style={{ fontSize: "0.6rem", fontWeight: 600, padding: "0.1rem 0.4rem", border: "1px solid var(--panel-border)", color: "var(--text-mute)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          [{regime.confidence}]
        </span>
      </div>

      {/* Description */}
      <p style={{ margin: "0 0 0.6rem 0", fontSize: "0.78rem", color: "var(--text-soft)", lineHeight: 1.5 }}>
        {regime.description}
      </p>

      {/* Signal grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.8rem", borderTop: "1px solid var(--panel-border)", paddingTop: "0.6rem" }}>
        {vix && vix.value != null && (
          <SignalBlock title="Volatility">
            <SignalRow label="VIX" value={vix.value} />
            <SignalRow label="Percentile" value={vix.percentile != null ? `${vix.percentile}%` : null} />
            <SignalRow label="Level" value={vix.level} />
          </SignalBlock>
        )}
        {dispersion && dispersion.value != null && (
          <SignalBlock title="Dispersion">
            <SignalRow label="Value" value={`${dispersion.value}%`} />
            <SignalRow label="Level" value={dispersion.level} />
          </SignalBlock>
        )}
        {correlations && (
          <SignalBlock title="Correlations">
            <SignalRow label="SPX/10Y" value={correlations.spx_vs_10y} />
            <SignalRow label="SPX/DXY" value={correlations.spx_vs_dxy} />
            <SignalRow label="Gold/10Y" value={correlations.gold_vs_10y} />
          </SignalBlock>
        )}
      </div>
    </div>
  );
}
