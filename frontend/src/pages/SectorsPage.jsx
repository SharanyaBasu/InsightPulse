import { useMarketData } from "../context/useMarketData";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import TerminalPanel from "../components/Terminal/TerminalPanel";
import PageHeader from "../components/Terminal/PageHeader";
import DataState from "../components/Terminal/DataState";
import HeatmapGrid from "../components/Terminal/HeatmapGrid";
export default function SectorsPage() {
  const { overview, loading, error, refresh, refreshing } = useMarketData();
  if (loading) return <TerminalLoader />;
  return <div className="page-stack">
    <PageHeader eyebrow="Market breadth" title="Sectors & regions" description="One-month performance. Signed returns show direction; shading shows the size of the move." />
    {error && <DataState error title="Market snapshot unavailable" message={overview ? "Showing the previous snapshot." : "Try again to load market performance."} onRetry={refresh} loading={refreshing} />}
    {overview && [["Regional performance · 1M", "regions", "region", 4], ["Sector rotation · 1M", "sectors", "sector", 5]].map(([title, key, label, columns]) => <TerminalPanel title={title} key={key}>{overview[key]?.length ? <HeatmapGrid items={overview[key]} labelKey={label} valueKey="change_1m" columns={columns} /> : <DataState title={`No ${key} data available`} />}</TerminalPanel>)}
  </div>;
}
