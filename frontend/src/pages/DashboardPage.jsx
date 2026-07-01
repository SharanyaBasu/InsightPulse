import { useState } from "react";
import { useMarketData } from "../context/MarketDataContext";
import useAssetHistory from "../hooks/useAssetHistory";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import TerminalPanel from "../components/Terminal/TerminalPanel";
import ChartModal from "../components/Terminal/ChartModal";
import Narrative from "../components/Overview/Narrative";
import MarketCard from "../components/Overview/MarketCard";
import RegimeLabel from "../components/Overview/RegimeLabel";
import YieldPanel from "../components/Overview/YieldPanel";
import MacroChips from "../components/Overview/MacroChips";
import CorrelationMonitor from "../components/Overview/CorrelationMonitor";
import RegionTile from "../components/Overview/RegionTile";
import SectorTile from "../components/Overview/SectorTile";

export default function DashboardPage() {
  const { overview, loading } = useMarketData();
  const [selectedAsset, setSelectedAsset] = useState(null);
  const { data: chartData } = useAssetHistory(selectedAsset?.symbol);

  if (loading) return <TerminalLoader />;
  if (!overview)
    return (
      <div style={{ padding: "1rem", color: "var(--red)" }}>
        ERROR: No data available
      </div>
    );

  const s = overview.sentiment || {};
  const scoreColor =
    s.score > 0
      ? "var(--green)"
      : s.score < 0
        ? "var(--red)"
        : "var(--text-soft)";

  // Top 8 movers from cross_asset by absolute 1D change
  const topMovers = overview.cross_asset
    ? [...overview.cross_asset]
        .filter((a) => a.change_1d != null)
        .sort((a, b) => Math.abs(b.change_1d) - Math.abs(a.change_1d))
        .slice(0, 8)
    : [];

  return (
    <div>
      {/* Sentiment Banner */}
      <TerminalPanel title="Sentiment" style={{ marginBottom: "0.5rem" }}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: "2rem",
            flexWrap: "wrap",
          }}
        >
          <div>
            <span
              style={{ color: scoreColor, fontSize: "1.6rem", fontWeight: 700 }}
            >
              {s.label}
            </span>
            <span
              style={{
                color: scoreColor,
                fontSize: "1.1rem",
                marginLeft: "0.6rem",
              }}
            >
              {s.score != null
                ? (s.score >= 0 ? "+" : "") + s.score.toFixed(2)
                : ""}
            </span>
          </div>
          <div style={{ fontSize: "0.85rem", color: "var(--text-soft)" }}>
            Equity Trend:{" "}
            <span style={{ fontWeight: 600, color: "var(--text)" }}>
              {s.equity_trend}
            </span>
          </div>
          <div style={{ fontSize: "0.85rem", color: "var(--text-soft)" }}>
            Drivers:{" "}
            <span style={{ color: "var(--text)" }}>
              {s.drivers?.join(", ")}
            </span>
          </div>
        </div>
      </TerminalPanel>

      {/* Narrative */}
      {overview.narrative && <Narrative text={overview.narrative} />}

      {/* Market Cards Strip */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "0.4rem",
          marginBottom: "0.5rem",
        }}
      >
        {overview.market_cards?.map((card) => (
          <MarketCard
            key={card.symbol}
            card={card}
            onClick={() => setSelectedAsset(card)}
          />
        ))}
      </div>

      {/* Two-column: Regime + Top Movers */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "0.5rem",
        }}
      >
        {/* Regime */}
        {overview.regime && (
          <TerminalPanel title="Macro Regime">
            <RegimeLabel regime={overview.regime} />
          </TerminalPanel>
        )}

        {/* Top Movers */}
        <TerminalPanel title="Top Movers (1D)">
          {topMovers.map((asset) => {
            const color =
              asset.change_1d > 0
                ? "var(--green)"
                : asset.change_1d < 0
                  ? "var(--red)"
                  : "var(--text-soft)";
            return (
              <div
                key={asset.symbol || asset.name}
                onClick={() => setSelectedAsset(asset)}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "0.3rem 0",
                  borderBottom: "1px solid var(--panel-border)",
                  fontSize: "0.85rem",
                  cursor: "pointer",
                }}
              >
                <span style={{ color: "var(--text)", fontWeight: 600 }}>
                  {asset.name}
                </span>
                <span
                  style={{
                    color,
                    fontWeight: 600,
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {asset.change_1d > 0 ? "+" : ""}
                  {asset.change_1d.toFixed(2)}%
                </span>
              </div>
            );
          })}
        </TerminalPanel>
      </div>

      {/* Rates + Macro */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "0.5rem",
          marginTop: "0.5rem",
        }}
      >
        {overview.yield && (
          <TerminalPanel title="Rates & Curve">
            <YieldPanel yieldData={overview.yield} />
          </TerminalPanel>
        )}
        {overview.macro && (
          <TerminalPanel title="Macro">
            <MacroChips macro={overview.macro} />
          </TerminalPanel>
        )}
      </div>

      {/* Cross-Asset Correlations */}
      {overview.correlations?.length > 0 && (
        <TerminalPanel
          title="Cross-Asset Correlations"
          style={{ marginTop: "0.5rem" }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: "0.5rem",
            }}
          >
            <CorrelationMonitor pairs={overview.correlations} />
          </div>
        </TerminalPanel>
      )}

      {/* Regions */}
      {overview.regions?.length > 0 && (
        <TerminalPanel title="Regions" style={{ marginTop: "0.5rem" }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: "0.4rem",
            }}
          >
            {overview.regions.map((r) => (
              <RegionTile key={r.symbol} tile={r} />
            ))}
          </div>
        </TerminalPanel>
      )}

      {/* Sectors */}
      {overview.sectors?.length > 0 && (
        <TerminalPanel title="Sectors" style={{ marginTop: "0.5rem" }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: "0.4rem",
            }}
          >
            {overview.sectors.map((s) => (
              <SectorTile key={s.symbol} tile={s} />
            ))}
          </div>
        </TerminalPanel>
      )}

      {/* Chart Modal */}
      {selectedAsset && chartData && (
        <ChartModal
          asset={selectedAsset}
          data={chartData}
          onClose={() => setSelectedAsset(null)}
        />
      )}
    </div>
  );
}
