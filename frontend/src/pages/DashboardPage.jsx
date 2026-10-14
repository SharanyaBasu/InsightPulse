import { useState } from "react";
import { Link } from "react-router-dom";
import { useMarketData } from "../context/useMarketData";
import useAssetHistory from "../hooks/useAssetHistory";
import useDailySummary from "../hooks/useDailySummary";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import TerminalPanel from "../components/Terminal/TerminalPanel";
import PageHeader from "../components/Terminal/PageHeader";
import DataState from "../components/Terminal/DataState";
import ChartModal from "../components/Terminal/ChartModal";
import MarketSummary from "../components/Overview/MarketSummary";
import MarketCard from "../components/Overview/MarketCard";
import RegimeLabel from "../components/Overview/RegimeLabel";
import YieldPanel from "../components/Overview/YieldPanel";
import MacroChips from "../components/Overview/MacroChips";
import CorrelationMonitor from "../components/Overview/CorrelationMonitor";
import RegionTile from "../components/Overview/RegionTile";
import SectorTile from "../components/Overview/SectorTile";

export default function DashboardPage() {
  const { overview, loading, error, refresh, refreshing } = useMarketData();
  const { summary, loading: summaryLoading, error: summaryError, refresh: refreshSummary } = useDailySummary();
  const [selectedAsset, setSelectedAsset] = useState(null);
  const chart = useAssetHistory(selectedAsset?.symbol);
  if (loading) return <TerminalLoader />;
  if (!overview) return <DataState error title="Market snapshot unavailable" message="We couldn't load the dashboard. Check your connection and try again." onRetry={refresh} loading={refreshing} />;
  const sentiment = overview.sentiment || {};
  const movers = [...(overview.cross_asset || [])].filter(asset => asset.change_1d != null).sort((a, b) => Math.abs(b.change_1d) - Math.abs(a.change_1d)).slice(0, 6);
  return <div className="page-stack">
    <PageHeader eyebrow="Market intelligence" title="The market, in perspective." description="Start with the takeaway. Explore the moves behind it."><Link className="ui-button" to="/scenario">Test a scenario ↗</Link></PageHeader>
    {error && <DataState error title="Refresh failed" message="Showing the previous market snapshot." onRetry={refresh} loading={refreshing} />}
    <div className="overview-lead">
      <MarketSummary summary={summary} loading={summaryLoading} error={summaryError} onRetry={refreshSummary} />
      <TerminalPanel title="Market pulse" className="pulse-panel">
        <span className="eyebrow">Sentiment</span><p className="pulse-label">{sentiment.label || "Unavailable"}</p>
        <p className="muted">Score <strong>{sentiment.score?.toFixed(2) ?? "—"}</strong> · Equity trend: {sentiment.equity_trend || "Unavailable"}</p>
        <p className="pulse-drivers">{sentiment.drivers?.join(" · ") || "No sentiment drivers available."}</p>
        <div className="pulse-regime"><h3>Macro regime</h3>{overview.regime ? <RegimeLabel regime={overview.regime} /> : <p className="muted">Regime data unavailable.</p>}</div>
      </TerminalPanel>
    </div>
    <div className="section-heading"><h2>Market snapshot</h2><span>Daily moves · Select a market to explore</span></div>
    {overview.market_cards?.length ? <div className="market-grid">{overview.market_cards.map(card => <MarketCard key={card.symbol} card={card} onClick={() => setSelectedAsset(card)} />)}</div> : <DataState title="No market quotes" message="Quotes will appear when the market feed has observations." />}
    <div className="section-heading"><h2>Behind the moves</h2><span>Macro context & market breadth</span></div>
    <div className="dashboard-grid">
      <TerminalPanel title="Largest daily moves">{movers.length ? movers.map(asset => <button className="mover-row" key={asset.symbol || asset.name} onClick={() => setSelectedAsset(asset)}><span>{asset.name}</span><strong style={{ color: asset.change_1d > 0 ? "var(--green)" : asset.change_1d < 0 ? "var(--red)" : "var(--text-soft)" }}>{asset.change_1d > 0 ? "+" : ""}{asset.change_1d.toFixed(2)}% <span aria-hidden="true">↗</span></strong></button>) : <DataState title="No daily movers available" />}</TerminalPanel>
      <TerminalPanel title="Rates & curve">{overview.yield ? <YieldPanel yieldData={overview.yield} /> : <DataState title="Rates unavailable" />}</TerminalPanel>
      <TerminalPanel title="Macro indicators">{overview.macro ? <MacroChips macro={overview.macro} /> : <DataState title="Macro indicators unavailable" />}</TerminalPanel>
      <TerminalPanel title="Cross-asset relationships" headerRight={<Link to="/cross-asset">Explore ↗</Link>}>{overview.correlations?.length ? <CorrelationMonitor pairs={overview.correlations} /> : <DataState title="No correlations available" />}</TerminalPanel>
    </div>
    <TerminalPanel title="Regional performance · 1M" headerRight={<Link to="/sectors">View breadth ↗</Link>}>
      {overview.regions?.length ? <div className="tile-grid">{overview.regions.map(tile => <RegionTile key={tile.symbol} tile={tile} />)}</div> : <DataState title="Regional data unavailable" />}
    </TerminalPanel>
    <TerminalPanel title="Sector performance · 1M">{overview.sectors?.length ? <div className="tile-grid">{overview.sectors.map(tile => <SectorTile key={tile.symbol} tile={tile} />)}</div> : <DataState title="Sector data unavailable" />}</TerminalPanel>
    {selectedAsset && <ChartModal key={selectedAsset.symbol} asset={selectedAsset} {...chart} onRetry={chart.refresh} onClose={() => setSelectedAsset(null)} />}
  </div>;
}
