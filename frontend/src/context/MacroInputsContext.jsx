import { useState, useCallback, useEffect, useRef } from "react";
import axios from "axios";

import { MACRO_INPUT_FIELDS, DEFAULT_MACRO_INPUTS, STORAGE_KEY, buildScenarioObject, loadSavedInputs, inputsMatch } from "./macroInputs";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

import { MacroInputsContext } from "./useMacroInputs";

export function MacroInputsProvider({ children }) {
  const [initialState] = useState(() => {
    try { return loadSavedInputs(window.localStorage); }
    catch { return { inputs: { ...DEFAULT_MACRO_INPUTS }, notice: null }; }
  });
  const [inputs, setInputs] = useState(initialState.inputs);
  const [unitNotice, setUnitNotice] = useState(initialState.notice);
  const [savedAt, setSavedAt] = useState(() => {
    try {
      return localStorage.getItem(`${STORAGE_KEY}.savedAt`);
    } catch {
      return null;
    }
  });
  const [dirty, setDirty] = useState(Boolean(initialState.notice));
  const [scenarioResult, setScenarioResult] = useState(null);
  const [lastRunAt, setLastRunAt] = useState(null);
  const [runLoading, setRunLoading] = useState(false);
  const [runError, setRunError] = useState(null);
  const [submittedInputs, setSubmittedInputs] = useState(null);
  const requestSequence = useRef(0);
  const activeRequest = useRef(null);
  const resultsStale = Boolean(submittedInputs && !inputsMatch(inputs, submittedInputs));

  useEffect(() => () => {
    requestSequence.current += 1;
    activeRequest.current?.abort();
  }, []);

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

    let parsed;
    if (typeof rawValue === "number") {
      parsed = rawValue;
    } else {
      const trimmed = String(rawValue).trim();
      if (trimmed === "" || trimmed === "-" || trimmed === "." || trimmed === "-.") {
        parsed = 0;
      } else {
        parsed = Number(trimmed);
        if (Number.isNaN(parsed)) return;
      }
    }

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
      setUnitNotice(null);
      setDirty(false);
    } catch (err) {
      console.error("Failed to save macro inputs:", err);
    }
  }, [inputs]);

  const resetInputs = useCallback(() => {
    requestSequence.current += 1;
    activeRequest.current?.abort();
    setRunLoading(false);
    setSubmittedInputs(null);
    setInputs(DEFAULT_MACRO_INPUTS);
    setScenarioResult(null);
    setLastRunAt(null);
    setRunError(null);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_MACRO_INPUTS));
      const stamp = new Date().toISOString();
      localStorage.setItem(`${STORAGE_KEY}.savedAt`, stamp);
      setSavedAt(stamp);
      setUnitNotice(null);
      setDirty(false);
    } catch {
      setDirty(false);
    }
  }, []);

  const runScenario = useCallback(async (overrideInputs) => {
    const source = { ...(overrideInputs || inputs) };
    const payload = buildScenarioObject(source);
    const sequence = ++requestSequence.current;
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setRunLoading(true);
    setRunError(null);
    try {
      const res = await axios.post("/api/scenario/run", payload, { signal: controller.signal });
      if (sequence !== requestSequence.current) return;
      setScenarioResult(res.data);
      setSubmittedInputs(source);
      setLastRunAt(new Date());
      return res.data;
    } catch (err) {
      if (sequence !== requestSequence.current || axios.isCancel(err)) return;
      const detail = err.response?.data?.detail;
      setRunError(typeof detail === "string" ? detail : err.message || "Scenario run failed");
      throw err;
    } finally {
      if (sequence === requestSequence.current) setRunLoading(false);
    }
  }, [inputs]);

  return (
    <MacroInputsContext.Provider
      value={{
        inputs,
        setInput,
        saveInputs,
        resetInputs,
        runScenario,
        scenarioResult,
        submittedInputs,
        resultsStale,
        lastRunAt,
        runLoading,
        runError,
        dirty,
        savedAt,
        unitNotice,
      }}
    >
      {children}
    </MacroInputsContext.Provider>
  );
}
