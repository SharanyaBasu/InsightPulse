/**
 * Frontend helpers for Scenario Playground result rendering.
 *
 * Canonical API contract lives in backend/schemas/scenario.py
 * Endpoint: POST /api/scenario/run
 *
 * Request body (ScenarioRunRequest):
 *   fed_funds_change_bps, cpi_surprise_pct, oil_change_pct, gdp_surprise_pct,
 *   unemployment_change_pct, pmi_change, dxy_change_pct, vix_change_pct
 *
 * Response body (ScenarioRunResponse):
 *   summary, regime, confidence, asset_deltas, sector_impacts, explanation
 *
 * GET /api/scenario is a separate live market delta endpoint and is not used here.
 */

const ASSET_META = {
  sp500: { label: "S&P 500", unit: "%" },
  nasdaq: { label: "NASDAQ", unit: "%" },
  ten_year_yield_bps: { label: "10Y Yield", unit: "bps" },
  dxy: { label: "DXY", unit: "%" },
  gold: { label: "Gold", unit: "%" },
  oil: { label: "Oil", unit: "%" },
};

const SECTOR_LABELS = {
  technology: "Technology",
  energy: "Energy",
  financials: "Financials",
  utilities: "Utilities",
  healthcare: "Healthcare",
  consumer_discretionary: "Consumer Disc.",
};

/** Map API asset_deltas object → ScenarioDeltaCards assets array */
export function toDeltaAssets(assetDeltas = {}) {
  return Object.entries(assetDeltas).map(([key, value]) => {
    const meta = ASSET_META[key] || { label: key, unit: "%" };
    return { label: meta.label, value, unit: meta.unit };
  });
}

/** Map API sector_impacts object → SectorHeatmap sectors array */
export function toHeatmapSectors(sectorImpacts = {}) {
  return Object.entries(sectorImpacts).map(([key, change]) => ({
    name: SECTOR_LABELS[key] || key,
    change,
  }));
}
