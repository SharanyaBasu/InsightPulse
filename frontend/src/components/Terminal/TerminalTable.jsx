import { useState } from "react";

export default function TerminalTable({ columns, data, onRowClick }) {
  const [sortCol, setSortCol] = useState(null);
  const [sortAsc, setSortAsc] = useState(true);

  function handleSort(colKey) {
    if (sortCol === colKey) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(colKey);
      setSortAsc(true);
    }
  }

  const sorted = sortCol
    ? [...data].sort((a, b) => {
        const av = a[sortCol];
        const bv = b[sortCol];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        const cmp = typeof av === "string" ? av.localeCompare(bv) : av - bv;
        return sortAsc ? cmp : -cmp;
      })
    : data;

  return (
    <table
      style={{
        width: "100%",
        borderCollapse: "collapse",
        fontSize: "0.78rem",
      }}
    >
      <thead>
        <tr>
          {columns.map((col) => (
            <th
              key={col.key}
              onClick={() => col.sortable !== false && handleSort(col.key)}
              style={{
                textAlign: col.align || "left",
                padding: "0.4rem 0.5rem",
                borderBottom: "1px solid var(--panel-border)",
                color: sortCol === col.key ? "var(--green)" : "var(--text-mute)",
                fontWeight: 600,
                fontSize: "0.65rem",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                cursor: col.sortable !== false ? "pointer" : "default",
                userSelect: "none",
                whiteSpace: "nowrap",
              }}
            >
              {col.label}
              {sortCol === col.key && (
                <span style={{ marginLeft: "0.3rem" }}>{sortAsc ? "▲" : "▼"}</span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map((row, i) => (
          <tr
            key={row._key || i}
            onClick={() => onRowClick?.(row)}
            style={{
              cursor: onRowClick ? "pointer" : "default",
              borderBottom: "1px solid rgba(255,255,255,0.03)",
            }}
          >
            {columns.map((col) => (
              <td
                key={col.key}
                style={{
                  padding: "0.35rem 0.5rem",
                  textAlign: col.align || "left",
                  color: "var(--text)",
                  whiteSpace: "nowrap",
                }}
              >
                {col.render ? col.render(row[col.key], row) : row[col.key]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
