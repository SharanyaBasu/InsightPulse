import { useState } from "react";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";
import Dialog from "./Dialog";
import DataState from "./DataState";
import ChartControls from "./ChartControls";
import ChartTooltip from "./ChartTooltip";
import { windowRows, assetUnit, shortDate, numberLabel } from "../../context/chartData";

export default function ChartModal({ asset, data = [], loading, error, onRetry, onClose }) {
  const [period, setPeriod] = useState(63);
  const rows = windowRows(data, period);
  const change = rows.length > 1 && rows[0].price !== 0 ? (rows.at(-1).price / rows[0].price - 1) * 100 : null;
  const unit = assetUnit(asset.symbol);
  return <Dialog title={`${asset.name || asset.symbol} · History`} onClose={onClose}>
    {error && rows.length > 0 && <DataState error title="History refresh failed" message="Showing previously loaded observations." onRetry={onRetry} loading={loading} />}
    {loading && !rows.length ? <DataState title="Loading price history…" /> : error && !rows.length ? <DataState error title="History unavailable" message="The history request failed. Try again." onRetry={onRetry} loading={loading} /> : !rows.length ? <DataState title="No history for this asset" message="This asset has no observations in the history feed yet." onRetry={onRetry} loading={loading} /> : <>
      <div className="chart-value"><strong>{numberLabel(rows.at(-1).price)}</strong><span>{unit}</span>{change != null && <span style={{ color: change > 0 ? "var(--green)" : change < 0 ? "var(--red)" : "var(--text-soft)" }}>{change > 0 ? "+" : ""}{change.toFixed(2)}% over displayed period</span>}</div>
      <ChartControls value={period} onChange={setPeriod} rows={rows} />
      <p className="muted">Closing observations · {unit}. Period buttons use 21, 63 or 252 observations.</p>
      <div className="chart-frame"><ResponsiveContainer width="100%" height="100%"><AreaChart data={rows} margin={{ right: 12, top: 12 }} accessibilityLayer>
        <CartesianGrid vertical={false} stroke="var(--panel-border)" />
        <XAxis dataKey="date" tickFormatter={shortDate} minTickGap={45} tick={{ fill: "var(--text-mute)", fontSize: 11 }} />
        <YAxis domain={["auto", "auto"]} width={70} tickFormatter={numberLabel} tick={{ fill: "var(--text-mute)", fontSize: 11 }} />
        <Tooltip content={<ChartTooltip unit={unit} />} />
        <Area dataKey="price" name={asset.name || asset.symbol} stroke="var(--green)" fill="var(--accent-wash)" strokeWidth={2} isAnimationActive={false} />
      </AreaChart></ResponsiveContainer></div>
    </>}
  </Dialog>;
}
