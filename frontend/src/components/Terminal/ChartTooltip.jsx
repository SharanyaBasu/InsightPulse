import { numberLabel } from "../../context/chartData";
export default function ChartTooltip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tooltip"><p>{label}</p>{payload.map(item => <div key={item.dataKey}><strong style={{ color: item.color }}>{item.name}</strong>: {numberLabel(item.value)} {unit}</div>)}</div>;
}
