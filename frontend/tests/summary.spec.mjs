import { test, expect } from '@playwright/test';

const overview = { market_cards: [], sentiment: {}, macro: { cpi: { value: 300 }, unemployment: { value: 4 }, policy_rate: { value: 5 } } };

test('summary errors are visible, retry recovers, and failed refresh keeps the previous summary', async ({ page }) => {
  let fail = true;
  await page.route('**/api/**', route => {
    if (route.request().url().endsWith('/overview')) return route.fulfill({ json: overview });
    if (route.request().url().endsWith('/summary/daily')) {
      return fail
        ? route.fulfill({ status: 503, json: { detail: 'A market summary is already being generated. Retry shortly.' } })
        : route.fulfill({ json: { headline: 'Evidence-backed test headline', validation: { status: 'verified' }, data_as_of: { oldest: '2026-09-01', latest: '2026-10-09' } } });
    }
    return route.fulfill({ json: [] });
  });
  await page.goto('/');
  await expect(page.getByText('A market summary is already being generated. Retry shortly.')).toBeVisible();
  await expect(page.getByText('Generating market summary…')).toHaveCount(0);
  fail = false;
  await page.getByRole('button', { name: 'Retry', exact: true }).click();
  await expect(page.getByText('Evidence-backed test headline')).toBeVisible();
  await expect(page.getByText('Underlying observations: 2026-09-01 – 2026-10-09')).toBeVisible();
  fail = true;
  await page.getByRole('button', { name: 'REFRESH', exact: true }).click();
  await expect(page.getByText(/Showing the previous summary/)).toBeVisible();
  await expect(page.getByText('Evidence-backed test headline')).toBeVisible();
});

test('repeated refresh clicks do not start overlapping summary requests', async ({ page }) => {
  let calls = 0;
  let hold = false;
  let finish;
  await page.route('**/api/**', async route => {
    if (route.request().url().endsWith('/overview')) return route.fulfill({ json: overview });
    if (route.request().url().endsWith('/summary/daily')) {
      calls += 1;
      if (hold) await new Promise(resolve => { finish = resolve; });
      return route.fulfill({ json: { headline: 'Cached test summary' } });
    }
    return route.fulfill({ json: [] });
  });
  await page.goto('/');
  await expect(page.getByText('Cached test summary')).toBeVisible();
  const beforeRefresh = calls;
  hold = true;
  const refresh = page.getByRole('button', { name: 'REFRESH', exact: true });
  await refresh.click();
  await expect(page.getByText('Checking for an updated summary…')).toBeVisible();
  await refresh.click();
  await refresh.click();
  await expect.poll(() => calls).toBe(beforeRefresh + 1);
  finish();
  await expect(page.getByText('Checking for an updated summary…')).toHaveCount(0);
});

test('an in-progress job is retried automatically using Retry-After', async ({ page }) => {
  let ready = false;
  await page.route('**/api/**', route => {
    if (route.request().url().endsWith('/overview')) return route.fulfill({ json: overview });
    if (route.request().url().endsWith('/summary/daily')) {
      return ready
        ? route.fulfill({ json: { headline: 'Completed background summary' } })
        : route.fulfill({ status: 503, headers: { 'Retry-After': '1' }, json: { detail: 'A market summary is already being generated. Retry shortly.' } });
    }
    return route.fulfill({ json: [] });
  });
  await page.goto('/');
  await expect(page.getByText('Generating market summary…', { exact: false })).toBeVisible();
  ready = true;
  await expect(page.getByText('Completed background summary')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Retry', exact: true })).toHaveCount(0);
});

test('supported claims reveal matching evidence and the verified badge is not a forecast', async ({ page }) => {
  const summary = {
    headline: 'The dollar strengthened',
    validation: { status: 'verified' },
    data_as_of: { oldest: '2026-06-10', latest: '2026-06-11' },
    regime_summary: {
      text: 'The dollar strengthened over five days.',
      evidence_ids: ['usd_index_5d_return'],
    },
    key_changes: [{ text: 'This line has no recorded indicator.', evidence_ids: [] }],
    risks_to_watch: [],
    mood_5d: { label: 'neutral', probability: 0.5, drivers: [] },
    limitations: ['Past moves can reverse.'],
    evidence: {
      usd_index_5d_return: {
        display_value: 0.8,
        unit: 'pct',
        window: '5d',
        timestamp: '2026-06-11T00:00:00',
        source: 'calculated_metrics',
        source_dependencies: ['market_data'],
      },
    },
  };
  await page.route('**/api/**', route => {
    if (route.request().url().endsWith('/overview')) return route.fulfill({ json: overview });
    if (route.request().url().endsWith('/summary/daily')) return route.fulfill({ json: summary });
    return route.fulfill({ json: [] });
  });
  await page.goto('/');
  await expect(page.getByText('Not individually sourced. The claims below carry the supporting indicators.')).toBeVisible();
  await expect(page.getByText("Checked against today's recorded indicators. Not a prediction or a guarantee.")).toBeVisible();
  await page.getByText('The dollar strengthened over five days.', { exact: true }).click();
  await expect(page.getByText('Dollar index, 5-day return')).toBeVisible();
  await expect(page.getByText('0.8%', { exact: true })).toBeVisible();
  await expect(page.getByText('5-day window', { exact: true })).toBeVisible();
  await expect(page.getByText('Observed 2026-06-11', { exact: true })).toBeVisible();
  await expect(page.getByText('Source: market_data', { exact: true })).toBeVisible();
  await page.getByText('Read the briefing', { exact: false }).click();
  await page.getByText('This line has no recorded indicator.', { exact: true }).click();
  await expect(page.getByText('No supporting evidence', { exact: true })).toBeVisible();
  await expect(page.getByText('Past moves can reverse.', { exact: true })).toBeVisible();
});
