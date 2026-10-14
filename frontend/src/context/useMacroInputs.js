import { createContext, useContext } from "react";

export const MacroInputsContext = createContext(null);

/** Access the shared MacroInputsContext provider state. */
export function useMacroInputs() {
  const ctx = useContext(MacroInputsContext);
  if (!ctx) throw new Error("useMacroInputs must be used within MacroInputsProvider");
  return ctx;
}
