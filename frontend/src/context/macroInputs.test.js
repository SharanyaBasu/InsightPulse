import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_MACRO_INPUTS, STORAGE_KEY, buildScenarioObject, loadSavedInputs } from './macroInputs.js';

const storage = (entries) => ({ getItem: (key) => entries[key] ?? null });

test('submitted shocks preserve percentage points, basis points, and VIX index points', () => {
  const payload = buildScenarioObject({ ...DEFAULT_MACRO_INPUTS, fedFundsRateChange: -25, cpiSurprise: 0.5, gdpGrowthSurprise: 0.5, unemploymentChange: 0.5, vixChangePoints: 5, oilPriceChange: 5, dxyChange: 1, pmiChange: 2 });
  assert.deepEqual(payload, { fed_funds_change_bps: -25, cpi_surprise_pct: 0.5, gdp_surprise_pct: 0.5, unemployment_change_pct: 0.5, vix_change_points: 5, oil_change_pct: 5, dxy_change_pct: 1, pmi_change: 2 });
});

test('legacy values retain other shocks but never reuse the ambiguous VIX value', () => {
  const result = loadSavedInputs(storage({ 'insightpulse.macroInputs': JSON.stringify({ vixChange: 25, unemploymentChange: 0.5, oilPriceChange: 5 }) }));
  assert.equal(result.inputs.vixChangePoints, 0);
  assert.equal(result.inputs.unemploymentChange, 0.5);
  assert.equal(result.inputs.oilPriceChange, 5);
  assert.match(result.notice, /reset to 0/);
});

test('versioned values take precedence over legacy storage and retain VIX points', () => {
  const result = loadSavedInputs(storage({ [STORAGE_KEY]: JSON.stringify({ vixChangePoints: 5 }), 'insightpulse.macroInputs': JSON.stringify({ vixChange: 25 }) }));
  assert.equal(result.inputs.vixChangePoints, 5);
  assert.equal(result.notice, null);
});

test('missing, invalid, or inaccessible storage safely restores neutral inputs', () => {
  for (const store of [storage({}), storage({ [STORAGE_KEY]: 'null' }), storage({ [STORAGE_KEY]: '{' }), { getItem() { throw new Error('blocked'); } }]) {
    assert.deepEqual(loadSavedInputs(store).inputs, DEFAULT_MACRO_INPUTS);
  }
});
