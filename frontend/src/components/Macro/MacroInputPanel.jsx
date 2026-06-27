import { useState, useCallback } from "react";
import {
  MACRO_INPUT_FIELDS,
  buildScenarioObject,
  useMacroInputs,
} from "../../context/MacroInputsContext";
import TerminalPanel from "../Terminal/TerminalPanel";

const inputStyle = {
  width: "100%",
  background: "var(--bg)",
  color: "var(--text)",
  border: "1px solid var(--panel-border)",
  borderRadius: "var(--radius)",
  fontSize: "0.85rem",
  fontFamily: "inherit",
  padding: "0.35rem 0.5rem",
  outline: "none",
  fontVariantNumeric: "tabular-nums",
};

const btnStyle = {
  background: "transparent",
  border: "1px solid var(--panel-border)",
  borderRadius: "var(--radius)",
  fontSize: "0.65rem",
  fontFamily: "inherit",
  padding: "0.15rem 0.45rem",
  cursor: "pointer",
  letterSpacing: "0.06em",
};

const PARTIAL_NUMBER = /^-?\d*\.?\d*$/;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function parseDraft(field, raw, fallback) {
  const trimmed = raw.trim();
  if (trimmed === "" || trimmed === "-" || trimmed === "." || trimmed === "-.") {
    return 0;
  }
  const parsed = Number(trimmed);
  if (Number.isNaN(parsed)) return fallback;
  return clamp(parsed, field.min, field.max);
}

export default function MacroInputPanel() {
  const { inputs, setInput, saveInputs, resetInputs, dirty, savedAt } = useMacroInputs();
  const [drafts, setDrafts] = useState({});
  const [runAt, setRunAt] = useState(null);
  const [runFlash, setRunFlash] = useState(false);

  const savedLabel = savedAt
    ? `Saved ${new Date(savedAt).toLocaleString()}`
    : "Not saved yet";

  const getDraft = useCallback(
    (field) => (drafts[field.key] !== undefined ? drafts[field.key] : String(inputs[field.key])),
    [drafts, inputs]
  );

  const getEffectiveInputs = useCallback(() => {
    const result = { ...inputs };
    for (const field of MACRO_INPUT_FIELDS) {
      if (drafts[field.key] !== undefined) {
        result[field.key] = parseDraft(field, drafts[field.key], inputs[field.key]);
      }
    }
    return result;
  }, [drafts, inputs]);

  const commitField = useCallback(
    (field) => {
      if (drafts[field.key] === undefined) return;
      const value = parseDraft(field, drafts[field.key], inputs[field.key]);
      setInput(field.key, value);
      setDrafts((prev) => {
        const next = { ...prev };
        delete next[field.key];
        return next;
      });
    },
    [drafts, inputs, setInput]
  );

  const handleFocus = (field) => {
    setDrafts((prev) => ({
      ...prev,
      [field.key]: inputs[field.key] === 0 ? "" : String(inputs[field.key]),
    }));
  };

  const handleChange = (field, value) => {
    if (value !== "" && !PARTIAL_NUMBER.test(value)) return;
    setDrafts((prev) => ({ ...prev, [field.key]: value }));
  };

  const handleBlur = (field) => {
    commitField(field);
  };

  const handleReset = () => {
    resetInputs();
    setDrafts({});
    setRunAt(null);
  };

  const handleRunScenario = () => {
    const effective = getEffectiveInputs();

    for (const field of MACRO_INPUT_FIELDS) {
      setInput(field.key, effective[field.key]);
    }
    setDrafts({});

    const scenario = buildScenarioObject(effective);

    console.log("Scenario:", scenario);
    setRunAt(new Date());
    setRunFlash(true);
    setTimeout(() => setRunFlash(false), 600);
  };

  return (
    <TerminalPanel
      title="Macro Scenario Inputs"
      headerRight={
        <div style={{ display: "flex", gap: "0.35rem", alignItems: "center" }}>
          <button
            type="button"
            onClick={saveInputs}
            disabled={!dirty}
            style={{
              ...btnStyle,
              color: dirty ? "var(--green)" : "var(--text-mute)",
              borderColor: dirty ? "var(--green)" : "var(--panel-border)",
              fontWeight: dirty ? 700 : 500,
              cursor: dirty ? "pointer" : "default",
            }}
          >
            SAVE
          </button>
          <button
            type="button"
            onClick={handleReset}
            style={{ ...btnStyle, color: "var(--text-mute)" }}
          >
            RESET
          </button>
        </div>
      }
      style={{ marginBottom: "0.5rem" }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
          gap: "0.75rem 1rem",
        }}
      >
        {MACRO_INPUT_FIELDS.map((field) => (
          <label
            key={field.key}
            style={{ display: "flex", flexDirection: "column", gap: "0.3rem", fontSize: "0.78rem" }}
          >
            <span style={{ color: "var(--text-soft)", lineHeight: 1.3 }}>
              {field.label}{" "}
              <span style={{ color: "var(--text-mute)" }}>({field.unit})</span>
            </span>
            <input
              type="text"
              inputMode="decimal"
              value={getDraft(field)}
              onFocus={() => handleFocus(field)}
              onChange={(e) => handleChange(field, e.target.value)}
              onBlur={() => handleBlur(field)}
              style={inputStyle}
            />
            <span style={{ fontSize: "0.65rem", color: "var(--text-mute)" }}>
              Range: {field.min} to {field.max} {field.unit}
            </span>
          </label>
        ))}
      </div>

      <div style={{ marginTop: "0.75rem", display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
        <button
          type="button"
          onClick={handleRunScenario}
          style={{
            ...btnStyle,
            padding: "0.4rem 0.85rem",
            fontSize: "0.72rem",
            fontWeight: 700,
            color: "#000",
            background: runFlash ? "var(--cyan)" : "var(--green)",
            borderColor: runFlash ? "var(--cyan)" : "var(--green)",
            letterSpacing: "0.08em",
            transition: "background 0.2s, border-color 0.2s",
          }}
        >
          RUN SCENARIO
        </button>
        {runAt && (
          <span style={{ fontSize: "0.72rem", color: "var(--green)", fontWeight: 600 }}>
            Scenario logged at {runAt.toLocaleTimeString()}
          </span>
        )}
      </div>

      <div style={{ marginTop: "0.6rem", fontSize: "0.65rem", color: dirty ? "var(--amber)" : "var(--text-mute)" }}>
        {dirty ? "Unsaved changes — click SAVE to keep these values after refresh." : savedLabel}
      </div>
    </TerminalPanel>
  );
}
