import { MACRO_INPUT_FIELDS, useMacroInputs } from "../../context/MacroInputsContext";
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

export default function MacroInputPanel() {
  const { inputs, setInput, saveInputs, resetInputs, dirty, savedAt } = useMacroInputs();

  const savedLabel = savedAt
    ? `Saved ${new Date(savedAt).toLocaleString()}`
    : "Not saved yet";

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
            onClick={resetInputs}
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
              type="number"
              min={field.min}
              max={field.max}
              step={field.step}
              value={inputs[field.key]}
              onChange={(e) => setInput(field.key, e.target.value)}
              style={inputStyle}
            />
            <span style={{ fontSize: "0.65rem", color: "var(--text-mute)" }}>
              Range: {field.min} to {field.max} {field.unit}
            </span>
          </label>
        ))}
      </div>
      <div style={{ marginTop: "0.6rem", fontSize: "0.65rem", color: dirty ? "var(--amber)" : "var(--text-mute)" }}>
        {dirty ? "Unsaved changes — click SAVE to keep these values after refresh." : savedLabel}
      </div>
    </TerminalPanel>
  );
}
