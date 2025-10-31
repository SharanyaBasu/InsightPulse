import { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts";

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get("/api/market-data")
      .then(res => setData(res.data))
      .catch(err => console.error(err));
  }, []);

  if (!data) return <p style={{ textAlign: "center" }}>Loading market data...</p>;

  const sentimentColor = data.sentiment === "bullish" ? "#00ff88" : "#ff5555";
  const sentimentText = data.sentiment.toUpperCase();

  const chartData = Object.entries(data.latest).map(([key, value]) => ({
    name: key,
    value
  }));

  return (
    <div className="container">
      <h1>InterMarket Insight</h1>

      <div className="card" style={{ textAlign: "center" }}>
        <h2 style={{ color: sentimentColor }}>
          Market Sentiment: {sentimentText}
        </h2>
        <p>Score: {data.score.toFixed(2)}</p>
      </div>

      <div className="card">
        <h3>Latest Prices</h3>
        <div className="grid">
          {Object.entries(data.latest).map(([k, v]) => (
            <div key={k}>
              <p className="value"><strong>{k}</strong></p>
              <p className="value">${v.toFixed(2)}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Market Overview</h3>
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
    </div>
  );
}

export default App;
