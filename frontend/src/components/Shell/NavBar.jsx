import { useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";

const TABS = [
  { key: "1", label: "OVERVIEW", to: "/" },
  { key: "2", label: "CROSS-ASSET", to: "/cross-asset" },
  { key: "3", label: "MACRO", to: "/macro" },
  { key: "4", label: "SECTORS", to: "/sectors" },
  { key: "5", label: "SIGNALS", to: "/signals" },
  { key: "6", label: "SCENARIO", to: "/scenario" },
];

export default function NavBar({ onShowHelp }) {
  const navigate = useNavigate();

  useEffect(() => {
    function handleKey(e) {
      // Don't intercept when typing in inputs
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

      const tab = TABS.find((t) => t.key === e.key);
      if (tab) {
        e.preventDefault();
        navigate(tab.to);
        return;
      }
      if (e.key === "?") {
        e.preventDefault();
        onShowHelp?.();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [navigate, onShowHelp]);

  return (
    <nav
      style={{
        display: "flex",
        alignItems: "center",
        gap: 0,
        background: "#0d0d0d",
        borderBottom: "1px solid var(--panel-border)",
        fontSize: "0.72rem",
        padding: "0 0.4rem",
      }}
    >
      {TABS.map((tab) => (
        <NavLink
          key={tab.key}
          to={tab.to}
          end={tab.to === "/"}
          style={({ isActive }) => ({
            display: "flex",
            alignItems: "center",
            gap: "0.3rem",
            padding: "0.45rem 0.8rem",
            color: isActive ? "#000" : "var(--text-soft)",
            background: isActive ? "var(--green)" : "transparent",
            fontWeight: isActive ? 700 : 500,
            textDecoration: "none",
            borderRight: "1px solid var(--panel-border)",
            transition: "background 0.1s, color 0.1s",
            letterSpacing: "0.04em",
          })}
        >
          <span className="t-kbd">
            {tab.key}
          </span>
          {tab.label}
        </NavLink>
      ))}

      <div style={{ marginLeft: "auto", padding: "0 0.8rem", color: "var(--text-mute)", fontSize: "0.65rem" }}>
        Press <span className="t-kbd">?</span> for shortcuts
      </div>
    </nav>
  );
}
