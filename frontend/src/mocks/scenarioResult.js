/**
 * Mock scenario response.
 *
 * Shape mirrors the contract the backend `POST /api/scenario` (or similar)
 * endpoint will eventually return. Keep this in sync with the backend so
 * frontend components can be developed against it now and swapped to a real
 * fetch later with no shape changes.
 *
 * Field reference:
 *   summary               One-sentence interpretation of the scenario.
 *   asset_deltas          Projected % change per major asset.
 *                         (ten_year_yield_bps is in basis points, not %.)
 *   sector_impacts        Projected % change per sector.
 *   explanation           Short paragraph explaining the drivers.
 */
export const MOCK_SCENARIO_RESULT = {
  summary: "This scenario suggests a mildly risk-off market environment.",
  asset_deltas: {
    sp500: -1.2,
    nasdaq: -1.8,
    ten_year_yield_bps: 15,
    dxy: 0.7,
    gold: 0.5,
    oil: 2.1,
  },
  sector_impacts: {
    technology: -2.0,
    energy: 1.4,
    financials: -0.6,
    utilities: 0.3,
    healthcare: 0.1,
    consumer_discretionary: -1.1,
  },
  explanation:
    "Higher VIX and oil prices usually create risk-off pressure. " +
    "Technology is negatively affected, while energy benefits from higher oil prices.",
};

export default MOCK_SCENARIO_RESULT;
