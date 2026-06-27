import { createContext, useContext, useState, useCallback, useEffect } from "react";

const STORAGE_KEY = "insightpulse.macroInputs";

export const MACRO_INPUT_FIELDS = [
  {
    key: "fedFundsRateChange",
    label: "Fed Funds Rate Change",
    unit: "bps",
    min: -100,
    max: 100,
    step: 1,
  },
  {
    key: "cpiSurprise",
    label: "CPI Surprise",
    unit: "%",
    min: -2,
    max: 2,
    step: 0.1,
  },
  {
    key: "oilPriceChange",
    label: "Oil Price Change",
    unit: "%",
    min: -30,
    max: 30,
    step: 0.5,
  },
  {
    key: "gdpGrowthSurprise",
    label: "GDP Growth Surprise",
    unit: "%",
    min: -5,
    max: 5,
    step: 0.1,
  },
  {
    key: "unemploymentChange",
    label: "Unemployment Change",
    unit: "%",
    min: -2,
    max: 2,
    step: 0.1,
  },
  {
    key: "pmiChange",
    label: "PMI Change",
    unit: "points",
    min: -10,
    max: 10,
    step: 0.5,
  },
  {
    key: "dxyChange",
    label: "DXY Change",
    unit: "%",
    min: -10,
    max: 10,
    step: 0.5,
  },
  {
    key: "vixChange",
    label: "VIX Change",
    unit: "%",
    min: -50,
    max: 100,
    step: 1,
  },
];

export const DEFAULT_MACRO_INPUTS = Object.fromEntries(
  MACRO_INPUT_FIELDS.map((f) => [f.key, 0])
);

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function loadSavedInputs() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_MACRO_INPUTS;
    const parsed = JSON.parse(raw);
    const merged = { ...DEFAULT_MACRO_INPUTS };
    for (const field of MACRO_INPUT_FIELDS) {
      const value = parsed[field.key];
      if (typeof value === "number" && !Number.isNaN(value)) {
        merged[field.key] = clamp(value, field.min, field.max);
      }
    }
    return merged;
  } catch {
    return DEFAULT_MACRO_INPUTS;
  }
}

const MacroInputsContext = createContext(null);

export function MacroInputsProvider({ children }) {
  const [inputs, setInputs] = useState(loadSavedInputs);
  const [savedAt, setSavedAt] = useState(() => {
    try {
      return localStorage.getItem(`${STORAGE_KEY}.savedAt`);
    } catch {
      return null;
    }
  });
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    const onBeforeUnload = (e) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [dirty]);

  const setInput = useCallback((key, rawValue) => {
    const field = MACRO_INPUT_FIELDS.find((f) => f.key === key);
    if (!field) return;

    const parsed = rawValue === "" ? 0 : Number(rawValue);
    if (Number.isNaN(parsed)) return;

    setInputs((prev) => ({
      ...prev,
      [key]: clamp(parsed, field.min, field.max),
    }));
    setDirty(true);
  }, []);

  const saveInputs = useCallback(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(inputs));
      const stamp = new Date().toISOString();
      localStorage.setItem(`${STORAGE_KEY}.savedAt`, stamp);
      setSavedAt(stamp);
      setDirty(false);
    } catch (err) {
      console.error("Failed to save macro inputs:", err);
    }
  }, [inputs]);

  const resetInputs = useCallback(() => {
    setInputs(DEFAULT_MACRO_INPUTS);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_MACRO_INPUTS));
      const stamp = new Date().toISOString();
      localStorage.setItem(`${STORAGE_KEY}.savedAt`, stamp);
      setSavedAt(stamp);
      setDirty(false);
    } catch {
      setDirty(false);
    }
  }, []);

  return (
    <MacroInputsContext.Provider
      value={{ inputs, setInput, saveInputs, resetInputs, dirty, savedAt }}
    >
      {children}
    </MacroInputsContext.Provider>
  );
}

export function useMacroInputs() {
  const ctx = useContext(MacroInputsContext);
  if (!ctx) throw new Error("useMacroInputs must be used within MacroInputsProvider");
  return ctx;
}
