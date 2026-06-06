import { useMarketData } from "../context/MarketDataContext";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import HeatmapGrid from "../components/Terminal/HeatmapGrid";

export default function SectorsPage() {
  const { overview, loading } = useMarketData();

  if (loading) return <TerminalLoader />;
  if (!overview) return <div style={{ color: "var(--red)", padding: "1rem" }}>NO DATA</div>;

  return (
    <div>
      {/* Regions */}
      <div style={{ marginBottom: "1rem" }}>
        <div style={{ fontSize: "0.8rem", color: "var(--text-mute)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "0.5rem", fontWeight: 600 }}>
          Regional Performance (1M)
        </div>
        {overview.regions && (
          <HeatmapGrid
            items={overview.regions}
            labelKey="region"
            valueKey="change_1m"
            columns={Math.min(overview.regions.length, 4)}
          />
        )}
      </div>

      {/* Sectors */}
      <div>
        <div style={{ fontSize: "0.8rem", color: "var(--text-mute)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "0.5rem", fontWeight: 600 }}>
          Sector Rotation (1M)
        </div>
        {overview.sectors && (
          <HeatmapGrid
            items={overview.sectors}
            labelKey="sector"
            valueKey="change_1m"
            columns={5}
          />
        )}
      </div>
    </div>
  );
}
