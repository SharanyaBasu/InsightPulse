import React from "react";
import { LineChart, Line, ResponsiveContainer } from "recharts";

export default function Sparkline({ data }) {
  const formatted = data.map((v, i) => ({ x: i, y: v }));

  return (
    <div style={{ width: "100%", height: 40 }}>
      <ResponsiveContainer>
        <LineChart data={formatted}>
          <Line
            type="monotone"
            dataKey="y"
            stroke="#4aa8ff"
            dot={false}
            strokeWidth={1.5}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
