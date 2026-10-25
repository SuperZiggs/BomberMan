import { expect } from '@playwright/test';
import { test, ready, openSetup, button, realGame, state, step, isPaused, currentTick } from './support';

test('menu → real match → pause, single step, reset and terminal results', async ({ page }, testInfo) => {
  const runtimeErrors: string[] = [];
  page.on('pageerror', e => runtimeErrors.push(e.message));
  const sid = await realGame(page);
  const first = await state(page, sid);
  expect(isPaused(first)).toBeTruthy();
  expect(currentTick(first)).toBe(0);
  const once = await step(page, sid);
  expect(isPaused(once)).toBeTruthy();
  await expect.poll(async () => currentTick(await state(page, sid))).toBe(1);
  await page.screenshot({ path: testInfo.outputPath('gameplay-desktop.png'), fullPage: true });
  await button(page, /Tiếp tục|Chạy|Bắt đầu|Resume/i).click();
  await expect.poll(async () => currentTick(await state(page, sid))).toBeGreaterThan(1);
  await button(page, /Tạm dừng|^Dừng$|Pause/i).click();
  await expect.poll(async () => isPaused(await state(page, sid))).toBeTruthy();
  const pausedTick = currentTick(await state(page, sid));
  await page.waitForTimeout(150);
  expect(currentTick(await state(page, sid))).toBe(pausedTick);
  await button(page, /Ván mới|Đặt lại|Chơi lại|Reset/i).click();
  await expect.poll(async () => currentTick(await state(page, sid))).toBe(0);
  const reset = await state(page, sid);
  expect(reset.match_id).not.toBe(first.match_id);
  expect(isPaused(reset)).toBeTruthy();
  for (let i = 0; i < 8 && !(await state(page, sid)).done; i++) await step(page, sid);
  expect((await state(page, sid)).done).toBeTruthy();
  await expect(page.locator('body')).toContainText(/Kết thúc|Kết quả|Hòa|chiến thắng|thắng trận/i);
  expect(runtimeErrors).toEqual([]);
});

test('keyboard tap, queued bomb and blur use authoritative engine ticks', async ({ page }) => {
  const sid = await realGame(page);
  await page.locator('canvas').first().click();
  await page.keyboard.press('ArrowRight');
  let frame = await step(page, sid);
  expect(frame.actions[0]).toBe(4);
  await page.locator('canvas').first().click();
  await page.keyboard.press('Space');
  await page.keyboard.press('KeyD');
  frame = await step(page, sid);
  expect(frame.actions[0]).toBe(5);
  await page.locator('canvas').first().click();
  await page.keyboard.down('ArrowDown');
  await page.evaluate(() => window.dispatchEvent(new Event('blur')));
  await page.keyboard.up('ArrowDown');
  frame = await step(page, sid);
  expect(frame.actions[0]).toBe(0);
});

test('recoverable model error and loading state are visible', async ({ page }) => {
  await page.route('**/api/models', route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Không thể tải checkpoint kiểm thử' }) }));
  await openSetup(page, 'play');
  await expect(page.locator('body')).toContainText(/lỗi|không thể|thất bại|thử lại/i);
  await page.unroute('**/api/models');
  await page.reload();
  await expect(page.locator('body')).toContainText(/Chơi|Huấn luyện/i);
});

test('connection state pauses after browser disconnect and restores full state', async ({ page, context }) => {
  const sid = await realGame(page);
  await button(page, /Tiếp tục|Chạy|Bắt đầu|Resume/i).click();
  await expect.poll(async () => currentTick(await state(page, sid))).toBeGreaterThan(0);
  await context.setOffline(true);
  await expect(page.locator('body')).toContainText(/Mất kết nối|Ngắt kết nối|Đang kết nối lại|Ngoại tuyến|Kết nối.*lỗi/i);
  await context.setOffline(false);
  await expect.poll(async () => isPaused(await state(page, sid))).toBeTruthy();
  await expect(page.locator('body')).toContainText(/Kết nối|Đã kết nối|Tạm dừng/i);
  expect((await state(page, sid)).session_id).toBe(sid);
});
