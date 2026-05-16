import { useState } from "react";
import { useMarketData } from "../context/MarketDataContext";
import useAssetHistory from "../hooks/useAssetHistory";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import CrossAssetPanel from "../components/Overview/CrossAssetPanel";
import ChartModal from "../components/Terminal/ChartModal";

export default function CrossAssetPage() {
  const { overview, loading } = useMarketData();
  const [filter, setFilter] = useState("");
  const [selectedAsset, setSelectedAsset] = useState(null);
  const { data: chartData } = useAssetHistory(selectedAsset?.symbol);

  if (loading) return <TerminalLoader />;
  if (!overview?.cross_asset) return <div style={{ color: "var(--red)", padding: "1rem" }}>NO DATA</div>;

  const filtered = filter
    ? overview.cross_asset.filter(
        (a) =>
          a.name?.toLowerCase().includes(filter.toLowerCase()) ||
          a.symbol?.toLowerCase().includes(filter.toLowerCase())
      )
    : overview.cross_asset;

  return (
    <div>
      {/* Header + Filter */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
        <div style={{ fontSize: "0.8rem", color: "var(--text-mute)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
          Cross-Asset Monitor — {filtered.length} instruments
          <span style={{ color: "var(--text-mute)", fontSize: "0.7rem", marginLeft: "1rem", fontStyle: "normal" }}>
            Click any row to expand chart
          </span>
        </div>
        <input
          type="text"
          placeholder="FILTER..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{
            background: "var(--panel)",
            border: "1px solid var(--panel-border)",
            borderRadius: "var(--radius)",
            color: "var(--text)",
            padding: "0.35rem 0.7rem",
            fontSize: "0.8rem",
            fontFamily: "inherit",
            outline: "none",
            width: 200,
          }}
        />
      </div>

      <CrossAssetPanel assets={filtered} onRowClick={(asset) => setSelectedAsset(asset)} />

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
