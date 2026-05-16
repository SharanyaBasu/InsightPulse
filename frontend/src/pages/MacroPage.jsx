import { useMarketData } from "../context/MarketDataContext";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import TerminalPanel from "../components/Terminal/TerminalPanel";
import YieldPanel from "../components/Overview/YieldPanel";
import MacroChips from "../components/Overview/MacroChips";
import RegimeLabel from "../components/Overview/RegimeLabel";
import CorrelationMonitor from "../components/Overview/CorrelationMonitor";

export default function MacroPage() {
  const { overview, loading } = useMarketData();

  if (loading) return <TerminalLoader />;
  if (!overview) return <div style={{ color: "var(--red)", padding: "1rem" }}>NO DATA</div>;

  return (
    <div>
      {/* Top: Yield + Macro side by side */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "0.5rem" }}>
        <TerminalPanel title="Yield Curve">
          {overview.yield && <YieldPanel yieldData={overview.yield} />}
        </TerminalPanel>
        <TerminalPanel title="Macro Indicators">
          {overview.macro && <MacroChips macro={overview.macro} />}
        </TerminalPanel>
      </div>

      {/* Regime */}
      {overview.regime && (
        <div style={{ marginBottom: "0.5rem" }}>
          <TerminalPanel title="Regime Classification">
            <RegimeLabel regime={overview.regime} />
          </TerminalPanel>
        </div>
      )}

      {/* Correlations */}
      {overview.correlations && (
        <>
          <div style={{ fontSize: "0.75rem", color: "var(--text-mute)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "0.4rem", fontWeight: 600 }}>
            Correlation Monitor
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "0.5rem" }}>
            {overview.correlations.map((pair) => (
              <CorrelationMonitor key={pair.key} pairs={[pair]} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
