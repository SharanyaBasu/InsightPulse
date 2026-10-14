import { useState } from "react";
import useHistory from "../hooks/useHistory";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import TerminalPanel from "../components/Terminal/TerminalPanel";
import DataState from "../components/Terminal/DataState";
import PageHeader from "../components/Terminal/PageHeader";
import ChartControls from "../components/Terminal/ChartControls";
import ChartTooltip from "../components/Terminal/ChartTooltip";
import { windowRows, normalizeComparison, assetUnit, shortDate, numberLabel } from "../context/chartData";
import { AreaChart, Area, LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid, ReferenceLine } from "recharts";
const SERIES = [{ key: "sp500", label: "S&P 500", color: "var(--green)" }, { key: "gold", label: "Gold", color: "var(--amber)", dash: "7 4" }, { key: "vix", label: "VIX", color: "var(--cyan)", dash: "2 4" }];
const tick = { fontSize: 11, fill: "var(--text-mute)" };
export default function SignalsPage() {
  const { history, loading, error, refresh } = useHistory();
  const [selectedAsset, setSelectedAsset] = useState("sp500");
  const [period, setPeriod] = useState(63);
  if (loading && !history) return <TerminalLoader message="LOADING HISTORY" />;
  const rows = windowRows(history || [], period);
  const assets = [...new Set((history || []).flatMap(row => Object.keys(row).filter(key => key !== "id" && key !== "date" && typeof row[key] === "number" && Number.isFinite(row[key]))))];
  const selected = assets.includes(selectedAsset) ? selectedAsset : assets[0];
  const prices = rows.filter(row => Number.isFinite(row[selected])).map(row => ({ date: row.date, price: row[selected] }));
  const series = SERIES.filter(item => rows.some(row => Number.isFinite(row[item.key])));
  const comparison = normalizeComparison(rows, series.map(item => item.key));
  return <div className="page-stack">
    <PageHeader eyebrow="Historical context" title="Signals & trends" description="Explore closing prices and compare relative moves over the same period." />
    {error && <DataState error title="History feed unavailable" message={history ? "Showing previously loaded history." : "Try again to load historical observations."} onRetry={refresh} loading={loading} />}
    {!history?.length ? !error && <DataState title="No history available" message="Historical charts need observations from the data feed." onRetry={refresh} loading={loading} /> : <>
      <ChartControls value={period} onChange={setPeriod} rows={rows} />
      <p className="muted">1M / 3M / 1Y use up to 21 / 63 / 252 observations. Dates above show the available window.</p>
      <TerminalPanel title="Historical prices" headerRight={<label className="inline-control">Asset<select aria-label="Asset" value={selected || ""} onChange={event => setSelectedAsset(event.target.value)}>{assets.map(asset => <option key={asset} value={asset}>{asset.toUpperCase()}</option>)}</select></label>}>
        <p className="muted">{selected?.toUpperCase()} · {assetUnit(selected)} · {prices.length ? `${prices[0].date} – ${prices.at(-1).date}` : "No observations"}</p>
        {prices.length ? <div className="chart-frame"><ResponsiveContainer width="100%" height="100%"><AreaChart data={prices} margin={{ right: 12, top: 12 }} accessibilityLayer>
          <CartesianGrid vertical={false} stroke="var(--panel-border)" /><XAxis dataKey="date" tickFormatter={shortDate} minTickGap={45} tick={tick} /><YAxis width={70} domain={["auto", "auto"]} tickFormatter={numberLabel} tick={tick} />
          <Tooltip content={<ChartTooltip unit={assetUnit(selected)} />} /><Area dataKey="price" name={selected?.toUpperCase()} stroke="var(--green)" fill="var(--accent-wash)" strokeWidth={2} isAnimationActive={false} />
        </AreaChart></ResponsiveContainer></div> : <DataState title="No observations for this asset in the selected period" />}
      </TerminalPanel>
      <TerminalPanel title="Relative performance · % change">
        <p className="muted">Each series starts at 0% on the first shared date. {comparison.length ? `${comparison[0].date} – ${comparison.at(-1).date}` : ""}</p>
        {series.length > 1 && comparison.length ? <>
          <div className="chart-frame"><ResponsiveContainer width="100%" height="100%"><LineChart data={comparison} margin={{ right: 12, top: 12 }} accessibilityLayer>
            <CartesianGrid vertical={false} stroke="var(--panel-border)" /><XAxis dataKey="date" tickFormatter={shortDate} minTickGap={45} tick={tick} /><YAxis width={70} tickFormatter={value => `${numberLabel(value)}%`} tick={tick} /><ReferenceLine y={0} stroke="var(--text-mute)" strokeDasharray="4 4" />
            <Tooltip content={<ChartTooltip unit="%" />} />{series.map(item => <Line key={item.key} dataKey={item.key} name={item.label} stroke={item.color} strokeDasharray={item.dash} dot={false} strokeWidth={2} isAnimationActive={false} />)}
          </LineChart></ResponsiveContainer></div>
          <div className="chart-legend">{series.map(item => <span key={item.key}><svg width="24" height="10" aria-hidden="true"><line x1="0" x2="24" y1="5" y2="5" stroke={item.color} strokeWidth="2" strokeDasharray={item.dash} /></svg>{item.label} <strong>{numberLabel(comparison.at(-1)[item.key])}%</strong></span>)}</div>
        </> : <DataState title="Not enough overlapping observations" message="At least two series with shared dates are needed for comparison." />}
        {series.length < SERIES.length && <p className="muted">Unavailable in this window: {SERIES.filter(item => !series.includes(item)).map(item => item.label).join(", ")}.</p>}
      </TerminalPanel>
    </>}
  </div>;
}
