import { test, expect } from '@playwright/test';

const snapshot = {
  market_cards: [],
  sentiment: {},
  macro: { cpi: { value: 300 }, unemployment: { value: 4 }, policy_rate: { value: 5 } },
  data_dates: { market: { sp500: '2026-06-11' } },
};

const contributionInputs = [
  ['fed_funds_change_bps', 'Fed Funds Rate Change', 'basis points'],
  ['cpi_surprise_pct', 'CPI Surprise', 'percentage points'],
  ['oil_change_pct', 'Oil Price Change', 'percent'],
  ['gdp_surprise_pct', 'GDP Growth Surprise', 'percentage points'],
  ['unemployment_change_pct', 'Unemployment Change', 'percentage points'],
  ['pmi_change', 'PMI Change', 'index points'],
  ['dxy_change_pct', 'DXY Change', 'percent'],
  ['vix_change_points', 'VIX Change', 'index points'],
];

function contributionDetail(unit = 'percent', displayedTotal = 0, values = {}, submitted = {}) {
  const contributions = contributionInputs.map(([input_key, label, input_unit]) => ({
    input_key,
    label,
    input_unit,
    submitted_value: submitted[input_key] || 0,
    contribution: values[input_key] || 0,
  }));
  const contributionSum = contributions.reduce((sum, row) => sum + row.contribution, 0);
  return {
    unit,
    unrounded_total: contributionSum,
    displayed_total: displayedTotal,
    contributions,
    contribution_sum: contributionSum,
    reconciliation_error: 0,
    reconciliation_tolerance: 1e-9,
    display_reconciliation_error: Math.abs(contributionSum - displayedTotal),
    display_rounding_tolerance: 0.005000001,
  };
}

const neutralContributions = {
  sp500: contributionDetail(),
  nasdaq: contributionDetail(),
  ten_year_yield_bps: contributionDetail('basis_points'),
  dxy: contributionDetail(),
  gold: contributionDetail(),
  oil: contributionDetail(),
};

const neutral = {
  summary: 'No shocks applied. Results match baseline. Zero means no additional impact, not a prediction that markets will stay flat.',
  regime: 'Low VIX', confidence: 'Low', explanation: 'Fixed historical state.',
  context: { baseline_date: '2026-06-11', horizon_trading_days: 21, effect_type: 'incremental_vs_baseline', neutral_inputs: true },
  asset_deltas: { sp500: 0, nasdaq: 0, ten_year_yield_bps: 0, dxy: 0, gold: 0, oil: 0 },
  asset_contributions: neutralContributions,
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
  const whyButtons = page.getByRole('button', { name: 'Why this result?' });
  await expect(whyButtons).toHaveCount(6);
  await whyButtons.nth(0).click();
  await expect(page.getByRole('heading', { name: 'S&P 500 model contributions' })).toBeVisible();
  await expect(page.getByText(/predictive associations, not proven causal effects/)).toBeVisible();
  await expect(page.getByText(/all submitted shocks were neutral/)).toBeVisible();
  expect(await page.locator('main').evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
  await whyButtons.nth(1).click();
  await expect(page.getByRole('heading', { name: 'NASDAQ model contributions' })).toBeVisible();
  await expect(whyButtons.nth(1)).toHaveAttribute('aria-expanded', 'true');
  await whyButtons.nth(1).click();
  await expect(page.getByRole('heading', { name: 'NASDAQ model contributions' })).toHaveCount(0);
});

test('failed rerun retains the original assumptions', async ({ page }) => {
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await expect(page.getByText(neutral.summary)).toBeVisible();
  await page.getByRole('button', { name: 'Why this result?' }).first().click();
  await page.getByRole('textbox', { name: /Oil Price Change/ }).fill('5');
  await page.route('**/api/scenario/run', route => route.fulfill({ status: 500, json: { detail: 'Test failure' } }));
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Latest run failed');
  await expect(page.getByRole('heading', { name: 'S&P 500 model contributions' })).toBeVisible();
  await page.getByText(/Displayed run:/).click();
  await expect(page.getByText('Oil Price Change: 0 %', { exact: true })).toBeVisible();
});

test('mixed model contributions render on both sides of zero', async ({ page }) => {
  const mixed = {
    ...neutral,
    summary: 'Mixed scenario contribution test.',
    context: { ...neutral.context, neutral_inputs: false },
    asset_deltas: { ...neutral.asset_deltas, sp500: -0.2 },
    asset_contributions: {
      ...neutral.asset_contributions,
      sp500: contributionDetail(
        'percent',
        -0.2,
        { fed_funds_change_bps: 0.4, vix_change_points: -0.6 },
        { fed_funds_change_bps: -25, vix_change_points: 10 },
      ),
    },
  };
  await page.route('**/api/scenario/run', route => route.fulfill({ json: mixed }));
  await page.getByRole('textbox', { name: /Fed Funds Rate Change/ }).fill('-25');
  await page.getByRole('textbox', { name: /VIX Change/ }).fill('10');
  await page.getByRole('button', { name: 'RUN SCENARIO', exact: true }).click();
  await page.getByRole('button', { name: 'Why this result?' }).first().click();

  const values = page.getByLabel('S&P 500 contribution values');
  await expect(values.getByText('+0.40 %', { exact: true })).toBeVisible();
  await expect(values.getByText('-0.60 %', { exact: true })).toBeVisible();
  await expect(page.locator('#asset-contribution-panel [fill="#1D9E75"]')).not.toHaveCount(0);
  await expect(page.locator('#asset-contribution-panel [fill="#D85A30"]')).not.toHaveCount(0);
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
    asset_deltas: { ...neutral.asset_deltas, oil: 12.34, ten_year_yield_bps: 7.89 },
    asset_contributions: {
      ...neutral.asset_contributions,
      oil: contributionDetail('percent', 12.34, { oil_change_pct: 12.34 }, { oil_change_pct: 25 }),
      ten_year_yield_bps: contributionDetail('basis_points', 7.89, { oil_change_pct: 7.89 }, { oil_change_pct: 25 }),
    },
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
  await expect(page.getByText('+12.34 %', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('+7.89 bps', { exact: true })).toBeVisible();
  const oilWhy = page.getByRole('button', { name: 'Why this result?' }).nth(5);
  await oilWhy.click();
  await expect(page.getByRole('heading', { name: 'Oil model contributions' })).toBeVisible();
  await expect(page.getByLabel('Oil contribution values').getByText('+12.34 %', { exact: true })).toBeVisible();
  await expect(page.getByText(/Contribution sum:/)).toBeVisible();
  await page.getByText(/Displayed run:/).click();
  await expect(page.getByText('Oil Price Change: 25 %', { exact: true })).toBeVisible();
});
