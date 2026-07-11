import { useMarketData } from "../context/MarketDataContext";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import TerminalPanel from "../components/Terminal/TerminalPanel";
import YieldPanel from "../components/Overview/YieldPanel";
import MacroChips from "../components/Overview/MacroChips";
import RegimeLabel from "../components/Overview/RegimeLabel";
import CorrelationMonitor from "../components/Overview/CorrelationMonitor";
import MacroInputPanel from "../components/Macro/MacroInputPanel";

export default function MacroPage() {
  const { overview, loading, error, refresh } = useMarketData();

  if (loading) return <TerminalLoader />;

  return (
    <div>
      <MacroInputPanel />

      {!overview && (
        <div style={{ color: "var(--red)", padding: "0 0 0.75rem 0", fontSize: "0.85rem" }}>
          Market data unavailable — check that the backend is running on port 8000.
          {error && (
            <span style={{ display: "block", color: "var(--text-mute)", marginTop: "0.25rem" }}>
              {error.message}
            </span>
          )}
          <button
            type="button"
            onClick={refresh}
            style={{
              display: "block",
              marginTop: "0.5rem",
              background: "transparent",
              color: "var(--cyan)",
              border: "1px solid var(--panel-border)",
              borderRadius: "var(--radius)",
              fontFamily: "inherit",
              fontSize: "0.72rem",
              padding: "0.25rem 0.5rem",
              cursor: "pointer",
            }}
          >
            RETRY
          </button>
        </div>
      )}

      {!overview ? null : (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "0.5rem" }}>
            <TerminalPanel title="Yield Curve">
              {overview.yield && <YieldPanel yieldData={overview.yield} />}
            </TerminalPanel>
            <TerminalPanel title="Macro Indicators">
              {overview.macro && <MacroChips macro={overview.macro} />}
            </TerminalPanel>
          </div>

          {overview.regime && (
            <div style={{ marginBottom: "0.5rem" }}>
              <TerminalPanel title="Regime Classification">
                <RegimeLabel regime={overview.regime} />
              </TerminalPanel>
            </div>
          )}

          {overview.correlations && (
            <>
              <div
                style={{
                  fontSize: "0.75rem",
                  color: "var(--text-mute)",
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  marginBottom: "0.4rem",
                  fontWeight: 600,
                }}
              >
                Correlation Monitor
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "0.5rem" }}>
                {overview.correlations.map((pair) => (
                  <CorrelationMonitor key={pair.key} pairs={[pair]} />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
