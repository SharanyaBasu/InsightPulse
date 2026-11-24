export const MACRO_INPUT_FIELDS = [
  {
    key: "fedFundsRateChange",
    example: "Rate from 5% to 4.75% = −25 bps.",
    label: "Fed Funds Rate Change",
    unit: "bps",
    min: -100,
    max: 100,
    step: 1,
  },
  {
    key: "cpiSurprise",
    example: "CPI at 3.5% versus 3% expected = +0.5 percentage points.",
    label: "CPI Surprise",
    unit: "percentage points",
    min: -2,
    max: 2,
    step: 0.1,
  },
  {
    key: "oilPriceChange",
    example: "Oil from $80 to $84 = +5%.",
    label: "Oil Price Change",
    unit: "%",
    min: -30,
    max: 30,
    step: 0.5,
  },
  {
    key: "gdpGrowthSurprise",
    example: "Growth at 2.5% versus 2% expected = +0.5 percentage points.",
    label: "GDP Growth Surprise",
    unit: "percentage points",
    min: -5,
    max: 5,
    step: 0.1,
  },
  {
    key: "unemploymentChange",
    example: "Unemployment from 4% to 4.5% = +0.5 percentage points.",
    label: "Unemployment Change",
    unit: "percentage points",
    min: -2,
    max: 2,
    step: 0.1,
  },
  {
    key: "pmiChange",
    example: "PMI from 50 to 52 = +2 index points.",
    label: "PMI Change",
    unit: "index points",
    min: -10,
    max: 10,
    step: 0.5,
  },
  {
    key: "dxyChange",
    example: "DXY from 100 to 101 = +1%.",
    label: "DXY Change",
    unit: "%",
    min: -10,
    max: 10,
    step: 0.5,
  },
  {
    key: "vixChangePoints",
    example: "VIX from 20 to 25 = +5 index points.",
    label: "VIX Change",
    unit: "index points",
    min: -50,
    max: 100,
    step: 1,
  },
];

export const DEFAULT_MACRO_INPUTS = Object.fromEntries(
  MACRO_INPUT_FIELDS.map((f) => [f.key, 0])
);

/** Map displayed shocks to the API without rescaling their units. */
export function buildScenarioObject(inputs) {
  return {
    fed_funds_change_bps: inputs.fedFundsRateChange,
    cpi_surprise_pct: inputs.cpiSurprise,
    oil_change_pct: inputs.oilPriceChange,
    gdp_surprise_pct: inputs.gdpGrowthSurprise,
    unemployment_change_pct: inputs.unemploymentChange,
    pmi_change: inputs.pmiChange,
    dxy_change_pct: inputs.dxyChange,
    vix_change_points: inputs.vixChangePoints,
  };
}

export const STORAGE_KEY = "insightpulse.macroInputs.v2";
const LEGACY_STORAGE_KEY = "insightpulse.macroInputs";

/** Load current inputs or migrate legacy values without reinterpreting VIX. */
export function loadSavedInputs(storage) {
  const empty = { inputs: { ...DEFAULT_MACRO_INPUTS }, notice: null };
  try {
    const current = storage.getItem(STORAGE_KEY);
    const legacy = current === null;
    const raw = current ?? storage.getItem(LEGACY_STORAGE_KEY);
    if (!raw) return empty;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return empty;
    const inputs = { ...DEFAULT_MACRO_INPUTS };
    for (const field of MACRO_INPUT_FIELDS) {
      if (legacy && field.key === "vixChangePoints") continue;
      const value = parsed[field.key];
      if (typeof value === "number" && Number.isFinite(value)) {
        inputs[field.key] = Math.min(field.max, Math.max(field.min, value));
      }
    }
    return {
      inputs,
      notice: legacy
        ? "VIX now uses index points. Your previously saved VIX input was reset to 0; other saved inputs were retained. Review VIX and save your inputs."
        : null,
    };
  } catch {
    return empty;
  }
}

/** Compare the current numeric assumptions with those used for a displayed result. */
export function inputsMatch(current, submitted) {
  return MACRO_INPUT_FIELDS.every(({ key }) => current[key] === submitted[key]);
}
