import { useState, useEffect, useCallback } from "react";
import { Outlet } from "react-router-dom";
import TopStatusBar from "./TopStatusBar";
import NavBar from "./NavBar";
import BottomStatusBar from "./BottomStatusBar";
import KeyboardHelp from "./KeyboardHelp";
import { useMarketData } from "../../context/useMarketData";

export default function TerminalShell() {
  const [showHelp, setShowHelp] = useState(false);
  const { refresh } = useMarketData();

  const toggleHelp = useCallback(() => setShowHelp((v) => !v), []);

  useEffect(() => {
    function handleKey(e) {
      if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest("input, textarea, select, [contenteditable], dialog")) return;
      if (e.key === "Escape") setShowHelp(false);
      if (e.key === "r") {
        e.preventDefault();
        refresh();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [refresh]);

  return (
    <div
      className="terminal-shell"
      style={{
        display: "grid",
        gridTemplateRows: "auto auto minmax(0, 1fr) auto",
        height: "100dvh",
        width: "100vw",
        overflow: "hidden",
        position: "relative",
      }}
    >
      <a className="skip-link" href="#main-content">Skip to content</a>
      <TopStatusBar />
      <NavBar onShowHelp={toggleHelp} />

      <main id="main-content" tabIndex={-1}
        style={{
          overflow: "auto",
          padding: "clamp(0.75rem, 2.4vw, 2rem)",
        }}
      >
        <Outlet />
      </main>

      <BottomStatusBar />

      {showHelp && <KeyboardHelp onClose={() => setShowHelp(false)} />}
    </div>
  );
}
