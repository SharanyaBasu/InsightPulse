import TerminalPanel from "../components/Terminal/TerminalPanel";
import ScenarioSummaryStrip from "../components/Scenario/ScenarioSummaryStrip";
import { MOCK_SCENARIO_RESULT } from "../mocks/scenarioResult";
import ScenarioDeltaCards from "../components/Overview/ScenarioDeltaCards";
import SectorHeatmap from "../components/Overview/SectorHeatmap";
import useScenario from "../hooks/useScenario";

export default function ScenarioPage() {
  const { data: scenario } = useScenario();
  const result = MOCK_SCENARIO_RESULT;

  return (
    <div>
      {/* Page header */}
      <div style={{ marginBottom: "0.75rem" }}>
        <h1 style={{
          fontSize: "1.3rem",
          fontWeight: 700,
          color: "var(--green)",
          letterSpacing: "0.04em",
          textTransform: "uppercase",
        }}>
          Scenario Playground
        </h1>
        <p style={{
          margin: "0.35rem 0 0 0",
          color: "var(--text-soft)",
          fontSize: "0.82rem",
          maxWidth: "65ch",
          lineHeight: 1.5,
        }}>
          Test &quot;what if&quot; macro scenarios. Adjust the macro inputs on the left
          (e.g. rate shocks, CPI surprises, oil moves) and see the projected market
          impact on the right.
        </p>
      </div>

      <div className="scenario-grid">
        <TerminalPanel title="Input Panel">
          <div style={{
            minHeight: 360,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--text-mute)",
            fontSize: "0.8rem",
            textAlign: "center",
            padding: "1rem",
          }}>
            Macro inputs will go here
            <br />
            (Fed Funds, CPI, Oil, GDP, Unemployment, PMI, DXY, VIX)
          </div>
        </TerminalPanel>

        <TerminalPanel title="Results">
          <ScenarioSummaryStrip summary={result?.summary} />
          {scenario && <ScenarioDeltaCards assets={scenario.assets} />}
          {scenario && <SectorHeatmap sectors={scenario.sectors} />}
        </TerminalPanel>
      </div>
    </div>
  );
}