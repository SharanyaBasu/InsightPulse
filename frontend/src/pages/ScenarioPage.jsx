import PageHeader from "../components/Terminal/PageHeader";
import { MACRO_INPUT_FIELDS } from "../context/macroInputs";
import MacroInputPanel from "../components/Macro/MacroInputPanel";
import TerminalPanel from "../components/Terminal/TerminalPanel";
import ScenarioSummaryStrip from "../components/Scenario/ScenarioSummaryStrip";
import ScenarioDeltaCards from "../components/Overview/ScenarioDeltaCards";
import SectorHeatmap from "../components/Overview/SectorHeatmap";
import { useMacroInputs } from "../context/useMacroInputs";
import { toDeltaAssets, toHeatmapSectors } from "../mocks/scenarioResult";

function ResultsPlaceholder() {
  return (
    <div
      style={{
        minHeight: 300,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "0.5rem",
        color: "var(--text-mute)",
        fontSize: "0.82rem",
        textAlign: "center",
        padding: "1.5rem 1rem",

      }}
    >
      <div style={{ fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", fontSize: "0.7rem" }}>
        Awaiting scenario
      </div>
      <div>
        Adjust the macro inputs, then click <span style={{ color: "var(--green)" }}>RUN SCENARIO</span>
        <br />
        to generate projected market impact.
      </div>
    </div>
  );
}

function RegimeConfidenceRow({ regime, confidence }) {
  if (!regime && !confidence) return null;
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.75rem",
        flexWrap: "wrap",
        marginBottom: "0.6rem",
        fontSize: "0.82rem",
      }}
    >
      {regime && (
        <span>
          <span style={{ color: "var(--text-mute)", marginRight: "0.35rem" }}>Regime</span>
          <span style={{ color: "var(--green)", fontWeight: 700 }}>{regime}</span>
        </span>
      )}
      {confidence && (
        <span
          style={{
            fontSize: "0.65rem",
            fontWeight: 600,
            padding: "0.1rem 0.4rem",
            border: "1px solid var(--panel-border)",
            color: "var(--text-mute)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
          }}
        >
          [{confidence}]
        </span>
      )}
    </div>
  );
}

function ExplanationBox({ text }) {
  if (!text) return null;
  return (
    <div
      style={{
        marginTop: "0.75rem",
        padding: "0.75rem 0.85rem",
        background: "rgba(0, 212, 255, 0.04)",
        border: "1px solid var(--panel-border)",
        borderLeft: "3px solid var(--cyan)",
        borderRadius: "var(--radius)",
      }}
    >
      <div
        style={{
          fontSize: "0.62rem",
          fontWeight: 700,
          color: "var(--cyan)",
          textTransform: "uppercase",
          letterSpacing: "0.14em",
          marginBottom: "0.35rem",
        }}
      >
        Explanation
      </div>
      <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--text)", lineHeight: 1.5 }}>
        {text}
      </p>
    </div>
  );
}

export default function ScenarioPage() {
  const { scenarioResult, runLoading, resultsStale, submittedInputs, lastRunAt, runError } = useMacroInputs();
  const hasResults = Boolean(scenarioResult);

  return (
    <div>
      <PageHeader eyebrow="Scenario analysis" title="Scenario Playground" description='Test "what if" macro scenarios. Adjust the inputs, run a scenario, and review the projected market impact.' />

      <div className="scenario-grid">
        <TerminalPanel title="Input Panel">
          <MacroInputPanel embedded />
        </TerminalPanel>

        <TerminalPanel
          title="Results"

        >
          {runLoading && !hasResults ? (
            <div style={{ color: "var(--text-mute)", padding: "2rem", textAlign: "center", fontSize: "0.85rem" }}>
              Running scenario…
            </div>
          ) : !hasResults ? (
            <ResultsPlaceholder />
          ) : (
            <>
              <div style={{ fontSize: "0.78rem", lineHeight: 1.6, marginBottom: "0.75rem" }}>
                {scenarioResult.context && (
                  <div style={{ padding: "0.65rem", border: "1px solid var(--panel-border)", marginBottom: "0.5rem" }}>
                    <strong>Historical baseline: {scenarioResult.context.baseline_date}</strong><br />
                    Horizon: {scenarioResult.context.horizon_trading_days} trading days.<br />
                    Values show additional impact relative to baseline, using historical—not live—market conditions.
                  </div>
                )}
                <div role="status" style={{ color: resultsStale || runError ? "var(--amber)" : "var(--text-soft)" }}>
                  {runLoading ? "Running scenario… Previous results remain below." : runError ? "Latest run failed. Previous results remain below." : resultsStale ? "Inputs changed—run again to update results." : "Results match the current inputs."}
                  {runLoading && resultsStale && " Inputs have changed since the displayed run."}
                </div>
                <details>
                  <summary>Displayed run: {lastRunAt?.toLocaleString()} — view assumptions</summary>
                  {submittedInputs && MACRO_INPUT_FIELDS.map((field) => (
                    <div key={field.key}>{field.label}: {submittedInputs[field.key]} {field.unit}</div>
                  ))}
                </details>
              </div>
              <ScenarioSummaryStrip summary={scenarioResult.summary} />
              <RegimeConfidenceRow
                regime={scenarioResult.regime}
                confidence={scenarioResult.confidence}
              />
              <ScenarioDeltaCards assets={toDeltaAssets(scenarioResult.asset_deltas)} />
              <SectorHeatmap scenario horizon={scenarioResult.context?.horizon_trading_days} sectors={toHeatmapSectors(scenarioResult.sector_impacts)} />
              <ExplanationBox text={scenarioResult.explanation} />
            </>
          )}
        </TerminalPanel>
      </div>
    </div>
  );
}
