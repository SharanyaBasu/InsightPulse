import MacroInputPanel from "../components/Macro/MacroInputPanel";
import TerminalPanel from "../components/Terminal/TerminalPanel";
import ScenarioSummaryStrip from "../components/Scenario/ScenarioSummaryStrip";
import ScenarioDeltaCards from "../components/Overview/ScenarioDeltaCards";
import SectorHeatmap from "../components/Overview/SectorHeatmap";
import { useMacroInputs } from "../context/MacroInputsContext";
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
        opacity: 0.55,
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
  const { scenarioResult, runLoading } = useMacroInputs();
  const hasResults = Boolean(scenarioResult);

  return (
    <div>
      <div style={{ marginBottom: "0.75rem" }}>
        <h1
          style={{
            fontSize: "1.3rem",
            fontWeight: 700,
            color: "var(--green)",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          Scenario Playground
        </h1>
        <p
          style={{
            margin: "0.35rem 0 0 0",
            color: "var(--text-soft)",
            fontSize: "0.82rem",
            maxWidth: "65ch",
            lineHeight: 1.5,
          }}
        >
          Test &quot;what if&quot; macro scenarios. Adjust the inputs, click Run Scenario,
          and review the projected market impact.
        </p>
      </div>

      <div className="scenario-grid">
        <TerminalPanel title="Input Panel">
          <MacroInputPanel embedded />
        </TerminalPanel>

        <TerminalPanel
          title="Results"
          style={{ opacity: hasResults || runLoading ? 1 : 0.7 }}
        >
          {runLoading && !hasResults ? (
            <div style={{ color: "var(--text-mute)", padding: "2rem", textAlign: "center", fontSize: "0.85rem" }}>
              Running scenario…
            </div>
          ) : !hasResults ? (
            <ResultsPlaceholder />
          ) : (
            <>
              <ScenarioSummaryStrip summary={scenarioResult.summary} />
              <RegimeConfidenceRow
                regime={scenarioResult.regime}
                confidence={scenarioResult.confidence}
              />
              <ScenarioDeltaCards assets={toDeltaAssets(scenarioResult.asset_deltas)} />
              <SectorHeatmap sectors={toHeatmapSectors(scenarioResult.sector_impacts)} />
              <ExplanationBox text={scenarioResult.explanation} />
            </>
          )}
        </TerminalPanel>
      </div>
    </div>
  );
}
