import { assetUnit, numberLabel } from "../../context/chartData";
export default function MarketCard({ card, onClick }) {
  const change = card.change_1d;
  const color = change > 0 ? "var(--green)" : change < 0 ? "var(--red)" : "var(--text-soft)";
  const series = (card.sparkline || []).filter(Number.isFinite);
  const low = Math.min(...series), spread = Math.max(...series) - low;
  const Tag = onClick ? "button" : "div";
  return <Tag className="market-card" onClick={onClick} aria-label={onClick ? `View ${card.name} history` : undefined}>
    <span className="card-label">{card.name}<span aria-hidden="true">↗</span></span>
    <span className="card-price">{card.price == null ? "—" : numberLabel(card.price)} <small>{assetUnit(card.symbol)}</small></span>
    <span className="card-movement"><strong style={{ color }}>{change == null ? "Unavailable" : `${change > 0 ? "+" : ""}${change.toFixed(2)}%`} <small>1D</small></strong>
      <svg aria-hidden="true" width="88" height="30" viewBox="0 0 88 30"><polyline fill="none" stroke={color} strokeWidth="1.5" points={series.map((v, i) => `${i / Math.max(1, series.length - 1) * 88},${spread ? 27 - (v - low) / spread * 24 : 15}`).join(" ")} /></svg>
    </span>
    <span className="card-periods">{[["1W", card.change_1w], ["1M", card.change_1m]].filter(([, value]) => value != null).map(([label, value]) => <span key={label}>{label} <b>{value > 0 ? "+" : ""}{value.toFixed(2)}%</b></span>)}</span>
    <span className="card-date">Observation: {card.observed_on || "date unavailable"}</span>
  </Tag>;
}
