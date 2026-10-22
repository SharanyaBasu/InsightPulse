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
  await page.getByText(/Displayed run:/).click();
  await expect(page.getByText('Oil Price Change: 0 %', { exact: true })).toBeVisible();
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

test('example presets can be inspected, edited, run, and reset', async ({ page }) => {
  await expect(page.getByText('Illustrative examples, not forecasts.')).toBeVisible();
  await expect(page.getByText('Active scenario: Custom', { exact: true })).toBeVisible();
  const presets = [
    { name: 'Rate cut', field: /Fed Funds Rate Change/, value: '-25', assumption: /Fed funds −25 bps/ },
    { name: 'Inflation surprise', field: /CPI Surprise/, value: '0.5', assumption: /CPI surprise \+0\.5 percentage points/ },
    { name: 'Oil shock', field: /Oil Price Change/, value: '10', assumption: /Oil price \+10%/ },
    { name: 'Growth slowdown', field: /GDP Growth Surprise/, value: '-0.5', assumption: /GDP growth −0\.5 percentage points/ },
  ];
  for (const preset of presets) {
    const button = page.getByRole('button', { name: preset.name });
    await expect(button).toContainText(preset.assumption);
    await expect(page.getByRole('textbox', { name: preset.field })).toHaveValue('0');
    await button.click();
    await expect(page.getByRole('textbox', { name: preset.field })).toHaveValue(preset.value);
    for (const other of presets) {
      if (other.name !== preset.name) await expect(page.getByRole('textbox', { name: other.field })).toHaveValue('0');
    }
    await expect(page.getByRole('textbox', { name: /VIX Change/ })).toHaveValue('0');
    await expect(button).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByText(`Active scenario: ${preset.name}`, { exact: true })).toBeVisible();
  }
  await page.getByRole('button', { name: 'Rate cut' }).click();
  await page.getByRole('textbox', { name: /Oil Price Change/ }).fill('5');
  await expect(page.getByText('Active scenario: Custom', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Rate cut' })).toHaveAttribute('aria-pressed', 'false');
  let submitted;
  await page.route('**/api/scenario/run', route => {
    submitted = route.request().postDataJSON();
    return route.fulfill({ json: neutral });
  });
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await expect(page.getByText(neutral.summary)).toBeVisible();
  expect(submitted).toEqual({ fed_funds_change_bps: -25, cpi_surprise_pct: 0, oil_change_pct: 5, gdp_surprise_pct: 0, unemployment_change_pct: 0, pmi_change: 0, dxy_change_pct: 0, vix_change_points: 0 });
  await page.getByRole('textbox', { name: /Oil Price Change/ }).fill('0');
  await expect(page.getByText('Active scenario: Rate cut', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'RESET', exact: true }).click();
  await expect(page.getByRole('textbox', { name: /Fed Funds Rate Change/ })).toHaveValue('0');
  await expect(page.getByText('Active scenario: Custom', { exact: true })).toBeVisible();
  await expect(page.getByText('Awaiting scenario', { exact: true })).toBeVisible();
});

test('changed scenario submits every input in canonical units and replaces neutral results', async ({ page }) => {
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await expect(page.getByText(neutral.summary)).toBeVisible();
  const changed = {
    ...neutral,
    summary: 'Changed oil shock projects additional oil and energy impact.',
    explanation: 'The displayed results use the submitted 25% oil shock.',
    context: { ...neutral.context, neutral_inputs: false },
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
  await page.getByText(/Displayed run:/).click();
  await expect(page.getByText('Oil Price Change: 25 %', { exact: true })).toBeVisible();
});
