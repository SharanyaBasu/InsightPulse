import { LineChart, Line, ResponsiveContainer } from "recharts";

export default function Sparkline({ data }) {
  const formatted = data.map((v, i) => ({ x: i, y: v }));

  return (
    <div style={{ width: "100%", height: 24 }}>
      <ResponsiveContainer>
        <LineChart data={formatted}>
          <Line type="monotone" dataKey="y" stroke="var(--green)" dot={false} strokeWidth={1.2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
