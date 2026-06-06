import { useState, useEffect, useCallback } from "react";
import { Outlet } from "react-router-dom";
import TopStatusBar from "./TopStatusBar";
import NavBar from "./NavBar";
import BottomStatusBar from "./BottomStatusBar";
import KeyboardHelp from "./KeyboardHelp";
import { useMarketData } from "../../context/MarketDataContext";

export default function TerminalShell() {
  const [showHelp, setShowHelp] = useState(false);
  const { refresh } = useMarketData();

  const toggleHelp = useCallback(() => setShowHelp((v) => !v), []);

  useEffect(() => {
    function handleKey(e) {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
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
      className="t-scanline"
      style={{
        display: "grid",
        gridTemplateRows: "auto auto 1fr auto",
        height: "100vh",
        width: "100vw",
        overflow: "hidden",
        position: "relative",
      }}
    >
      <TopStatusBar />
      <NavBar onShowHelp={toggleHelp} />

      <main
        style={{
          overflow: "auto",
          padding: "0.5rem 0.6rem",
        }}
      >
        <Outlet />
      </main>

      <BottomStatusBar />

      {showHelp && <KeyboardHelp onClose={() => setShowHelp(false)} />}
    </div>
  );
}
