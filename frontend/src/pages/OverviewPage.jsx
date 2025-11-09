import React from "react";
import useOverview from "../hooks/useOverview";

// Components
import Narrative from "../components/Overview/Narrative";
import MarketCard from "../components/Overview/MarketCard";
import SectorTile from "../components/Overview/SectorTile";
import RegionTile from "../components/Overview/RegionTile";
import MacroChips from "../components/Overview/MacroChips";
import YieldPanel from "../components/Overview/YieldPanel";

export default function OverviewPage() {
  const { data, loading } = useOverview();

  if (loading) return <p style={{ padding: "2rem" }}>Loading...</p>;
  if (!data) return <p style={{ padding: "2rem" }}>Error loading data.</p>;

  return (
    <div style={{ padding: "2rem", maxWidth: "1500px", margin: "0 auto" }}>
      {/* PAGE HEADER */}
      <div style={{ marginBottom: "2rem" }}>
        <h1
          style={{ fontSize: "2.6rem", fontWeight: 700, color: "var(--blue)" }}
        >
          Market Overview
        </h1>
      </div>

      <div
        style={{
          background: "#111",
          padding: "1rem 1.5rem",
          borderRadius: "12px",
          marginBottom: "1.5rem",
          color: "#fff",
          border: "1px solid #222",
        }}
      >
        <h2 style={{ margin: 0, color: "#4cc9f0" }}>
          Sentiment: {data.sentiment.label} (Score:{" "}
          {data.sentiment.score.toFixed(2)})
        </h2>

        <p style={{ margin: "0.5rem 0 0 0", opacity: 0.8 }}>
          Equity Trend: {data.sentiment.equity_trend}
        </p>

        <p style={{ margin: "0.5rem 0 0 0", opacity: 0.8 }}>
          Drivers: {data.sentiment.drivers.join(", ")}
        </p>
      </div>

      {/* NARRATIVE STRIP */}
      <div
        style={{
          background: "var(--panel)",
          padding: "1.2rem 1.6rem",
          borderRadius: "10px",
          marginBottom: "2rem",
          border: "1px solid var(--border)",
          boxShadow: "0 0 20px rgba(0,0,0,0.15)",
        }}
      >
        <Narrative text={data.narrative} />
      </div>

      {/* MARKET STRIP (Bloomberg-style wide band) */}
      <div
        style={{
          display: "flex",
          gap: "1rem",
          overflowX: "auto",
          paddingBottom: "1rem",
          marginBottom: "3rem",
        }}
      >
        {data.market_cards.map((card) => (
          <div
            key={card.symbol}
            style={{
              background: "var(--card)",
              minWidth: "200px",
              padding: "1rem",
              borderRadius: "10px",
              border: "1px solid var(--border)",
            }}
          >
            <MarketCard card={card} />
          </div>
        ))}
      </div>

      {/* YIELD + MACRO GRID */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "2rem",
          marginBottom: "3rem",
        }}
      >
        <div
          style={{
            background: "var(--panel)",
            padding: "1.6rem",
            borderRadius: "12px",
            border: "1px solid var(--border)",
          }}
        >
          <YieldPanel yieldData={data.yield} />
        </div>

        <div
          style={{
            background: "var(--panel)",
            padding: "1.6rem",
            borderRadius: "12px",
            border: "1px solid var(--border)",
          }}
        >
          <MacroChips macro={data.macro} />
        </div>
      </div>

      {/* REGIONS */}
      <h2 style={{ color: "var(--blue)", marginBottom: "1rem" }}>Regions</h2>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "1.2rem",
          marginBottom: "3rem",
        }}
      >
        {data.regions.map((r) => (
          <RegionTile key={r.symbol} tile={r} />
        ))}
      </div>

      {/* SECTORS */}
      <h2 style={{ color: "var(--blue)", marginBottom: "1rem" }}>Sectors</h2>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "1.2rem",
          marginBottom: "3rem",
        }}
      >
        {data.sectors.map((s) => (
          <SectorTile key={s.symbol} tile={s} />
        ))}
      </div>
    </div>
  );
}
