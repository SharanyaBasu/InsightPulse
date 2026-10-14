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
