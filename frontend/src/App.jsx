import { BrowserRouter, Routes, Route } from "react-router-dom";
import { MarketDataProvider } from "./context/MarketDataContext";
import { MacroInputsProvider } from "./context/MacroInputsContext";
import TerminalShell from "./components/Shell/TerminalShell";
import DashboardPage from "./pages/DashboardPage";
import CrossAssetPage from "./pages/CrossAssetPage";
import MacroPage from "./pages/MacroPage";
import SectorsPage from "./pages/SectorsPage";
import SignalsPage from "./pages/SignalsPage";
import ScenarioPage from "./pages/ScenarioPage";

export default function App() {
  return (
    <MarketDataProvider>
      <MacroInputsProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<TerminalShell />}>
            <Route index element={<DashboardPage />} />
            <Route path="cross-asset" element={<CrossAssetPage />} />
            <Route path="macro" element={<MacroPage />} />
            <Route path="sectors" element={<SectorsPage />} />
            <Route path="signals" element={<SignalsPage />} />
            <Route path="scenario" element={<ScenarioPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
      </MacroInputsProvider>
    </MarketDataProvider>
  );
}
