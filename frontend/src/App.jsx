import { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

function App() {
  const [data, setData] = useState(null);
  const [history, setHistory] = useState([]);

  // Fetch snapshot data (sentiment + latest prices)
  useEffect(() => {
    axios
      .get("/api/market-data")
      .then((res) => setData(res.data))
      .catch((err) => console.error(err));
  }, []);

  // Fetch historical time series
  useEffect(() => {
    axios
      .get("/api/history")
      .then((res) => setHistory(res.data))
      .catch((err) => console.error(err));
  }, []);

  if (!data)
    return <p style={{ textAlign: "center" }}>Loading market data...</p>;

  const sentimentColor = data.sentiment === "bullish" ? "#00ff88" : "#ff5555";
  const sentimentText = data.sentiment.toUpperCase();

  const chartData = Object.entries(data.latest).map(([key, value]) => ({
    name: key,
    value,
  }));

  return (
    <div
      style={{
        backgroundColor: "#0a0a0a",
        color: "#ffffff",
        minHeight: "100vh",
        padding: "2rem",
        fontFamily: "Inter, sans-serif",
      }}
    >
      <h1 style={{ textAlign: "center", marginBottom: "2rem" }}>
        InterMarket Insight 📊
      </h1>

      {/* SENTIMENT CARD */}
      <div
        style={{
          textAlign: "center",
          marginBottom: "2rem",
          background: "#111",
          padding: "1.5rem",
          borderRadius: "10px",
        }}
      >
        <h2 style={{ color: sentimentColor }}>
          Market Sentiment: {sentimentText}
        </h2>
        <p>Score: {data.score.toFixed(2)}</p>
      </div>

      {/* LATEST PRICES GRID */}
      <div
        style={{
          background: "#111",
          padding: "1.5rem",
          borderRadius: "10px",
          marginBottom: "2rem",
        }}
      >
        <h3 style={{ marginBottom: "1rem" }}>Latest Prices</h3>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
            gap: "1rem",
          }}
        >
          {Object.entries(data.latest).map(([k, v]) => (
            <div key={k} style={{ textAlign: "center" }}>
              <p style={{ fontWeight: "bold", color: "#58a6ff" }}>{k}</p>
              <p>${v.toFixed(2)}</p>
            </div>
          ))}
        </div>
      </div>

      {/* MARKET SNAPSHOT CHART */}
      <div
        style={{
          background: "#111",
          padding: "1.5rem",
          borderRadius: "10px",
          marginBottom: "2rem",
        }}
      >
        <h3 style={{ marginBottom: "1rem" }}>Market Overview</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <XAxis dataKey="name" stroke="#8b949e" />
            <YAxis stroke="#8b949e" />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#58a6ff"
              strokeWidth={2}
              dot={{ fill: "#58a6ff" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* HISTORICAL MARKET TREND CHART */}
      <div
        style={{
          background: "#111",
          padding: "1.5rem",
          borderRadius: "10px",
        }}
      >
        <h3 style={{ marginBottom: "1rem" }}>Historical Market Trends</h3>
        {history.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
              <XAxis dataKey="date" tick={{ fill: "#aaa", fontSize: 10 }} />
              <YAxis tick={{ fill: "#aaa" }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1e1e1e", border: "none" }}
                labelStyle={{ color: "#ccc" }}
              />
              <Legend />
              <Line type="monotone" dataKey="sp500" stroke="#00b4d8" dot={false} />
              <Line type="monotone" dataKey="nasdaq" stroke="#90e0ef" dot={false} />
              <Line type="monotone" dataKey="gold" stroke="#ffd60a" dot={false} />
              <Line type="monotone" dataKey="oil" stroke="#ff6b6b" dot={false} />
              <Line
                type="monotone"
                dataKey="usd_index"
                stroke="#6a4c93"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="ten_year_yield"
                stroke="#4cc9f0"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p style={{ textAlign: "center" }}>Loading historical data...</p>
        )}
      </div>
    </div>
  );
}

export default App;
