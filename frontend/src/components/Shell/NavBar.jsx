import { useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";

const TABS = [
  { key: "1", label: "OVERVIEW", to: "/" },
  { key: "2", label: "CROSS-ASSET", to: "/cross-asset" },
  { key: "3", label: "SECTORS", to: "/sectors" },
  { key: "4", label: "SIGNALS", to: "/signals" },
  { key: "5", label: "SCENARIO", to: "/scenario" },
];

export default function NavBar({ onShowHelp }) {
  const navigate = useNavigate();

  useEffect(() => {
    function handleKey(e) {
      // Don't intercept when typing in inputs
      if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest("input, textarea, select, [contenteditable], dialog")) return;

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
    <nav className="main-nav" aria-label="Main navigation">
      {TABS.map((tab) => (
        <NavLink
          key={tab.key}
          to={tab.to}
          end={tab.to === "/"}
          className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
        >
          <span className="t-kbd">
            {tab.key}
          </span>
          {tab.label}
        </NavLink>
      ))}

      <button className="help-button" onClick={onShowHelp} aria-label="Keyboard shortcuts"><span className="t-kbd">?</span> Shortcuts</button>
    </nav>
  );
}
