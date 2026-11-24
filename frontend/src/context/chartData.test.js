import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeComparison, windowRows, assetUnit } from './chartData.js';

test('comparison aligns dates, drops unusable observations and uses a shared baseline', () => {
  const rows = [
    { date: '2026-01-01', sp500: 5000, gold: null },
    { date: '2026-01-02', sp500: 5000, gold: 2000 },
    { date: '2026-01-03', sp500: 5500, gold: 1800 },
    { date: '2026-01-04', sp500: 0, gold: 1800 },
    { date: '2026-01-05', sp500: NaN, gold: 1800 },
  ];
  const result = normalizeComparison(rows, ['sp500', 'gold']);
  assert.equal(result.length, 2);
  assert.deepEqual(result[0], { date: '2026-01-02', sp500: 0, gold: 0 });
  assert.ok(Math.abs(result[1].sp500 - 10) < 1e-10);
  assert.ok(Math.abs(result[1].gold + 10) < 1e-10);
  assert.deepEqual(normalizeComparison(rows, ['vix']), []);
});

test('windows select available trailing observations and units distinguish rates, FX and prices', () => {
  assert.deepEqual(windowRows([1, 2, 3], 2), [2, 3]);
  assert.deepEqual(windowRows([1, 2, 3], 0), [1, 2, 3]);
  assert.equal(assetUnit('tnx'), '% yield');
  assert.equal(assetUnit('sp500'), 'index points');
  assert.equal(assetUnit('usdjpy'), 'JPY per USD');
  assert.equal(assetUnit('gold'), 'USD / troy oz');
  assert.equal(assetUnit('unknown'), 'quote units');
});
