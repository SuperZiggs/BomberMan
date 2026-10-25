import { expect, type Page } from '@playwright/test';
import { test, openSetup, field, choose, button, state, step, currentTick, isPaused } from './support';

const mapLabel = /Kích thước.*(?:bản đồ|bàn|map)|Cỡ bản đồ|Map size|Board size/i;
const sizeLabel = (size: number) => new RegExp(`${size}\\s*[×xX]\\s*${size}`);

async function chooseMap(page: Page, size: number) {
  const control = page.getByRole('combobox', { name: mapLabel })
    .or(page.getByRole('radiogroup', { name: mapLabel }))
    .or(page.getByRole('group', { name: mapLabel }));
  await expect(control, 'Match setup must expose a labelled map-size choice').toBeVisible();
  await expect(control).toBeEnabled();
  // A visible label is required in addition to an accessible name.
  await expect(page.getByText(mapLabel).filter({ visible: true }).first()).toBeVisible();
  if (await control.evaluate(el => el.tagName.toLowerCase()) === 'select') {
    for (const value of [9, 11]) await expect(control.locator('option').filter({ hasText: sizeLabel(value) })).toHaveCount(1);
    const selected = control.locator('option:checked');
    await expect(selected).toHaveText(sizeLabel(11));
    const value = await control.locator('option').filter({ hasText: sizeLabel(size) }).getAttribute('value');
    await control.selectOption(value!);
  } else {
    for (const value of [9, 11]) await expect(control.getByRole('radio', { name: sizeLabel(value) })).toBeVisible();
    await expect(control.getByRole('radio', { name: sizeLabel(11) })).toBeChecked();
    await control.getByRole('radio', { name: sizeLabel(size) }).check();
  }
}

for (const mode of ['play', 'watch'] as const) for (const size of [9, 11]) {
  test(`${mode}: choose ${size}x${size}, render, single-step and reset real match`, async ({ page }, testInfo) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/sessions', async route => {
      if (route.request().method() !== 'POST') return route.continue();
      // Bound the real match; do not supply or overwrite the board-size choice.
      await route.continue({ postData: JSON.stringify({ ...route.request().postDataJSON(), max_steps: 8, seed: 912790 }) });
    });
    await openSetup(page, mode);
    await chooseMap(page, size);
    await choose(field(page, /Số tác tử|Agents/i), mode === 'play' ? '3' : '2');
    const creation = page.waitForResponse(r => /\/api\/sessions$/.test(new URL(r.url()).pathname) && r.request().method() === 'POST');
    await button(page, /Bắt đầu|Vào trận|Tạo ván|Start/i).click();
    const response = await creation;
    expect(response.request().postDataJSON().board_size).toBe(size);
    expect(response.ok(), await response.text()).toBeTruthy();
    const { session_id: sid, snapshot: first } = await response.json();
    expect(first.grid).toHaveLength(size);
    expect(first.grid.every((row: number[]) => row.length === size)).toBeTruthy();
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible();
    // Existing terrain sprites are 32px cells; the bitmap must cover the actual grid.
    await expect(canvas).toHaveAttribute('width', String(size * 32));
    await expect(canvas).toHaveAttribute('height', String(size * 32));
    const once = await step(page, sid);
    expect(once.grid).toHaveLength(size);
    await button(page, /Ván mới|Đặt lại|Chơi lại|Reset/i).click();
    await expect.poll(async () => currentTick(await state(page, sid))).toBe(0);
    const reset = await state(page, sid);
    expect(reset.match_id).not.toBe(first.match_id);
    expect(isPaused(reset)).toBeTruthy();
    expect(reset.grid).toHaveLength(size);
    expect(reset.grid.every(row => row.length === size)).toBeTruthy();
    await expect(canvas).toHaveAttribute('width', String(size * 32));
    await expect(canvas).toHaveAttribute('height', String(size * 32));
    await page.screenshot({ path: testInfo.outputPath(`${mode}-${size}x${size}.png`), fullPage: true });
    expect(errors).toEqual([]);
  });
}
