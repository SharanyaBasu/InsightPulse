import { CHART_WINDOWS } from "../../context/chartData";
export default function ChartControls({ value, onChange, rows }) {
  return <div className="chart-controls"><div className="range-buttons" role="group" aria-label="Chart period">
    {CHART_WINDOWS.map(item => <button key={item.label} className="ui-button" aria-pressed={value === item.value} onClick={() => onChange(item.value)}>{item.label}</button>)}
    </div><span className="muted">{rows.length ? `${rows[0].date} – ${rows.at(-1).date} · ${rows.length} observations` : "No observations"}</span></div>;
}
