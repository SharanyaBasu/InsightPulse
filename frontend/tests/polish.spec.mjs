import { test, expect } from '@playwright/test';
const history = Array.from({ length: 80 }, (_, i) => ({ date: new Date(Date.UTC(2026, 0, i + 1)).toISOString().slice(0, 10), sp500: 5000 + i * 10, gold: 2000 - i * 2, tnx: 4 + i / 1000 }));
const overview = {
  market_cards: [{ symbol: 'sp500', name: 'S&P 500', price: 5790, change_1d: 0.3, change_1w: 1.5, change_1m: 4.2, sparkline: [1, 1.05, 1.02, 1.08], observed_on: '2026-03-21' }, { symbol: 'vgk', name: 'Europe', price: 85, change_1d: null }],
  sentiment: { label: 'Neutral', score: 0, equity_trend: 'Mixed', drivers: ['Equities up', 'Gold down'] },
  macro: {}, regions: [{ region: 'US', change_1m: 4.2 }], sectors: [{ sector: 'Technology', change_1m: 5 }],
};
async function mock(page) {
  await page.route('**/api/**', route => route.fulfill({ json: route.request().url().endsWith('/overview') ? overview : route.request().url().endsWith('/summary/daily') ? { headline: 'Equities rise as gold retreats', regime_summary: 'Mixed conditions across markets.', key_changes: ['Equities gained.'] } : history }));
}

test('all pages fit desktop and mobile; help remains keyboard accessible', async ({ page }) => {
  await mock(page);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const width of [1440, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    for (const path of ['/', '/cross-asset', '/sectors', '/signals', '/scenario']) {
      await page.goto(path);
      await expect(page.locator('main h1')).toBeVisible();
      await expect.poll(() => page.locator('main').evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
    }
  }
  await page.getByRole('button', { name: 'Keyboard shortcuts' }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  for (let i = 0; i < 5; i++) await page.keyboard.press('Tab');
  await expect.poll(() => page.evaluate(() => Boolean(document.activeElement.closest('dialog')))).toBe(true);
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible();
  await expect(page.getByRole('button', { name: 'Keyboard shortcuts' })).toBeFocused();
  expect(errors).toEqual([]);
});

test('history dialog opens from keyboard, reports failure, retries and restores focus', async ({ page }) => {
  await mock(page);
  let fail = true;
  await page.route('**/api/history', route => fail ? route.fulfill({ status: 500, json: {} }) : route.fulfill({ json: history }));
  await page.goto('/');
  const card = page.getByRole('button', { name: 'View S&P 500 history' });
  await card.focus();
  await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog');
  await expect(dialog.getByText('History unavailable', { exact: true })).toBeVisible();
  fail = false;
  await dialog.getByRole('button', { name: 'Retry' }).click();
  await expect(dialog.getByText(/over displayed period/)).toBeVisible();
  await expect(dialog.getByRole('button', { name: '3M', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await dialog.getByRole('button', { name: '1M', exact: true }).click();
  await expect(dialog.getByText(/21 observations/)).toBeVisible();
  await page.keyboard.press('1');
  await expect(dialog).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(card).toBeFocused();
  await page.getByRole('button', { name: 'View Europe history' }).click();
  await expect(page.getByRole('dialog').getByText('No history for this asset', { exact: true })).toBeVisible();
});

test('overview retry recovers and failed refresh preserves quotes', async ({ page }) => {
  await mock(page);
  let fail = true;
  await page.route('**/api/overview', route => fail ? route.fulfill({ status: 500, json: {} }) : route.fulfill({ json: overview }));
  await page.goto('/');
  await expect(page.getByText('Market snapshot unavailable', { exact: true })).toBeVisible();
  fail = false;
  await page.getByRole('button', { name: 'Retry', exact: true }).click();
  await expect(page.getByRole('button', { name: 'View S&P 500 history' })).toBeVisible();
  fail = true;
  await page.locator('main').focus();
  await page.keyboard.press('r');
  await expect(page.getByText('Showing the previous market snapshot.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'View S&P 500 history' })).toBeVisible();
});

test('signals recovers from history failure, normalizes comparison, and protects select input', async ({ page }) => {
  await mock(page);
  let fail = true;
  await page.route('**/api/history', route => fail ? route.fulfill({ status: 500, json: {} }) : route.fulfill({ json: history }));
  await page.goto('/signals');
  await expect(page.getByText('History feed unavailable')).toBeVisible();
  fail = false;
  await page.getByRole('button', { name: 'Retry', exact: true }).click();
  await expect(page.getByText(/Each series starts at 0%/)).toBeVisible();
  await expect(page.getByText('Unavailable in this window: VIX.')).toBeVisible();
  await page.getByLabel('Asset', { exact: true }).focus();
  await page.keyboard.press('1');
  await expect(page).toHaveURL(/\/signals$/);
  await page.getByRole('button', { name: '1M', exact: true }).click();
  await expect(page.getByText(/21 observations/)).toBeVisible();
});

test('cross-asset empty pair keeps controls available to recover', async ({ page }) => {
  await mock(page);
  const sparse = history.map((row, i) => ({ ...row, gold: i === 0 ? 2000 : null }));
  await page.route('**/api/history', route => route.fulfill({ json: sparse }));
  await page.goto('/cross-asset');
  await expect(page.getByText('NO OVERLAPPING HISTORY')).toBeVisible();
  await page.getByLabel('Comparison asset').selectOption('tnx');
  await expect(page.getByRole('heading', { name: 'Normalized performance' })).toBeVisible();
});
