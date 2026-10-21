import { test, expect } from '@playwright/test';

const snapshot = {
  market_cards: [],
  sentiment: {},
  macro: { cpi: { value: 300 }, unemployment: { value: 4 }, policy_rate: { value: 5 } },
  data_dates: { market: { sp500: '2026-06-11' } },
};
const neutral = {
  summary: 'No shocks applied. Results match baseline. Zero means no additional impact, not a prediction that markets will stay flat.',
  regime: 'Low VIX', confidence: 'Low', explanation: 'Fixed historical state.',
  context: { baseline_date: '2026-06-11', horizon_trading_days: 21, effect_type: 'incremental_vs_baseline', neutral_inputs: true },
  warnings: [{ code: 'STATE_SNAPSHOT_STALE', severity: 'high', message: 'The default state snapshot is not current relative to the diagnostic date.' }],
  calculation_details: {
    model_version: 'scenario-playground-all-target-ridge-v1',
    bundle_version: 'scenario-playground-ridge-v1.0.0',
    baseline_date: '2026-06-11',
    horizon_trading_days: 21,
    submitted_inputs: {
      oil_change_pct: { label: 'Oil Price Change', value: 0, unit: 'percent' },
      vix_change_points: { label: 'VIX Change', value: 0, unit: 'index points' },
    },
    targets: {
      sp500: { label: 'S&P 500', unit: 'percent', baseline_prediction: 1.2345, scenario_prediction: 1.2345, difference: 0 },
      oil: { label: 'Oil', unit: 'percent', baseline_prediction: 1, scenario_prediction: 1, difference: 0 },
    },
  },
  asset_deltas: { sp500: 0, nasdaq: 0, ten_year_yield_bps: 0, dxy: 0, gold: 0, oil: 0 },
  sector_impacts: { technology: 0, energy: 0, financials: 0, utilities: 0, healthcare: 0, consumer_discretionary: 0 },
};

test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', route => {
    if (route.request().url().endsWith('/overview')) return route.fulfill({ json: snapshot });
    if (route.request().url().endsWith('/scenario/run')) return route.fulfill({ json: neutral });
    return route.fulfill({ json: [] });
  });
  await page.goto('/scenario');
});

test('neutral rendering, historical context and edits before blur', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await expect(page.getByText(neutral.summary)).toBeVisible();
  await expect(page.getByText('Historical baseline: 2026-06-11')).toBeVisible();
  await expect(page.getByText(/Horizon: 21 trading days/)).toBeVisible();
  await expect(page.getByRole('alert', { name: 'Historical state warnings' })).toContainText('STATE_SNAPSHOT_STALE');
  await expect(page.getByText(/^\+0(?:\.00)?\s*(?:%|bps)$/)).toHaveCount(0);
  const oil = page.getByRole('textbox', { name: /Oil Price Change/ });
  await oil.fill('5');
  await expect(page.getByRole('status')).toContainText('Inputs changed');
  await oil.fill('0');
  await expect(page.getByRole('status')).toContainText('Results match the current inputs');
  await expect(page.locator('main')).not.toContainText('NaN');
  expect(await page.locator('main').evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
  await expect(page.locator('main svg rect').first()).toHaveCSS('fill', 'rgb(52, 59, 70)');
});

test('failed rerun retains the original assumptions', async ({ page }) => {
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await expect(page.getByText(neutral.summary)).toBeVisible();
  await page.getByRole('textbox', { name: /Oil Price Change/ }).fill('5');
  await page.route('**/api/scenario/run', route => route.fulfill({ status: 500, json: { detail: 'Test failure' } }));
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Latest run failed');
  await page.getByText('Calculation details', { exact: true }).click();
  await expect(page.getByText('0.00 %', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('scenario-playground-all-target-ridge-v1', { exact: true })).toBeVisible();
});

test('edits during a request remain stale; reset prevents a late response restoring results', async ({ page }) => {
  let finish;
  let started;
  let startedPromise = new Promise(resolve => { started = resolve; });
  await page.route('**/api/scenario/run', async route => {
    const wait = new Promise(resolve => { finish = resolve; });
    started();
    await wait;
    await route.fulfill({ json: neutral }).catch(() => {});
  });
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await startedPromise;
  await page.getByRole('textbox', { name: /Oil Price Change/ }).fill('5');
  finish();
  await expect(page.getByRole('status')).toContainText('Inputs changed');
  startedPromise = new Promise(resolve => { started = resolve; });
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await startedPromise;
  await page.getByRole('button', { name: 'RESET', exact: true }).click();
  finish();
  await expect(page.getByText('Awaiting scenario', { exact: true })).toBeVisible();
  await expect(page.getByText(neutral.summary)).toHaveCount(0);
});

test('refresh timestamps do not imply that old observations are live', async ({ page }) => {
  await expect(page.getByText('Market data as of 2026-06-11')).toBeVisible();
  await expect(page.getByText('CONNECTED', { exact: true })).toBeVisible();
  await expect(page.getByText(/Last checked:/)).toBeVisible();
  await page.route('**/api/overview', route => route.fulfill({ status: 500, json: { detail: 'offline' } }));
  await page.keyboard.press('r');
  await expect(page.getByText('REFRESH FAILED', { exact: true })).toBeVisible();
  await expect(page.getByText('Market data as of 2026-06-11')).toBeVisible();
  await page.route('**/api/overview', route => route.fulfill({ json: snapshot }));
  await page.keyboard.press('r');
  await expect(page.getByText('CONNECTED', { exact: true })).toBeVisible();
  await expect(page.getByText('Market data as of 2026-06-11')).toBeVisible();
});

test('changed scenario submits every input in canonical units and replaces neutral results', async ({ page }) => {
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await expect(page.getByText(neutral.summary)).toBeVisible();
  const changed = {
    ...neutral,
    summary: 'Changed oil shock projects additional oil and energy impact.',
    explanation: 'The displayed results use the submitted 25% oil shock.',
    context: { ...neutral.context, neutral_inputs: false },
    calculation_details: {
      ...neutral.calculation_details,
      submitted_inputs: {
        ...neutral.calculation_details.submitted_inputs,
        oil_change_pct: { label: 'Oil Price Change', value: 25, unit: 'percent' },
      },
      targets: {
        ...neutral.calculation_details.targets,
        oil: { label: 'Oil', unit: 'percent', baseline_prediction: 1, scenario_prediction: 13.34, difference: 12.34 },
      },
    },
    asset_deltas: { ...neutral.asset_deltas, oil: 12.34, ten_year_yield_bps: 7.89 },
    sector_impacts: { ...neutral.sector_impacts, energy: 4.56 },
  };
  let submitted;
  await page.route('**/api/scenario/run', route => {
    submitted = route.request().postDataJSON();
    return route.fulfill({ json: changed });
  });
  await page.getByRole('textbox', { name: /Oil Price Change/ }).fill('25');
  await expect(page.getByRole('status')).toContainText('Inputs changed');
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await expect(page.getByText(changed.summary)).toBeVisible();
  expect(submitted).toEqual({ fed_funds_change_bps: 0, cpi_surprise_pct: 0, oil_change_pct: 25, gdp_surprise_pct: 0, unemployment_change_pct: 0, pmi_change: 0, dxy_change_pct: 0, vix_change_points: 0 });
  await expect(page.getByRole('status')).toContainText('Results match the current inputs');
  await expect(page.getByText(neutral.summary)).toHaveCount(0);
  await expect(page.getByText('+12.34 %', { exact: true })).toBeVisible();
  await expect(page.getByText('+7.89 bps', { exact: true })).toBeVisible();
  await page.getByText('Calculation details', { exact: true }).click();
  await expect(page.getByText('25.00 %', { exact: true })).toBeVisible();
  await expect(page.getByRole('row', { name: /Oil 1.00 % 13.34 % 12.34 %/ })).toBeVisible();
});
