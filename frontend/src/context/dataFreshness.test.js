import { test } from 'node:test';
import assert from 'node:assert/strict';
import { observationRange, connectionStatus } from './dataFreshness.js';
import { inputsMatch, DEFAULT_MACRO_INPUTS } from './macroInputs.js';

test('successful requests do not replace old observation dates', () => {
  const overview = { data_dates: { market: { sp500: '2026-06-11', gold: '2026-06-10' } } };
  assert.equal(connectionStatus({ overview }), 'CONNECTED');
  assert.equal(observationRange(overview.data_dates.market), '2026-06-10 – 2026-06-11');
  assert.equal(connectionStatus({ overview, error: new Error('offline') }), 'REFRESH FAILED');
  assert.equal(connectionStatus({ error: new Error('offline') }), 'UNAVAILABLE');
  assert.equal(observationRange({ cpi: null }), 'Unknown');
});

test('stale results compare values and clear when the submitted values are restored', () => {
  const submitted = { ...DEFAULT_MACRO_INPUTS };
  assert.equal(inputsMatch({ ...submitted, oilPriceChange: 5 }, submitted), false);
  assert.equal(inputsMatch({ ...submitted }, submitted), true);
});
