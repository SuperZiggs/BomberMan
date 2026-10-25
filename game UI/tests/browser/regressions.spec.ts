import { expect } from '@playwright/test';
import { test, button, realGame, state, step, openSetup, field, choose } from './support';

test('loading catalog cannot overwrite a chosen three-agent setup', async ({ page }) => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/models', async route => { await gate; await route.continue(); });
  await openSetup(page, 'play');
  const count = field(page, /Số tác tử/);
  try { await expect(count).toBeDisabled(); } finally { release(); }
  await expect(count).toBeEnabled();
  await choose(count, '3');
  const created = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/api/sessions'));
  await button(page, /Bắt đầu ván/).click();
  expect((await (await created).json()).snapshot.agents).toHaveLength(3);
});

test('toolbar pause and reset clear client-held directions', async ({ page }) => {
  const sid = await realGame(page, 'play', 3, 100);
  await page.locator('canvas').first().click();
  await page.keyboard.down('ArrowRight');
  await button(page, /Tiếp tục/).click();
  await expect.poll(async () => (await state(page, sid)).tick).toBeGreaterThan(0);
  await button(page, /Tạm dừng/).click();
  await page.keyboard.press('Space');
  expect((await step(page, sid)).actions[0]).toBe(5);
  expect((await step(page, sid)).actions[0]).toBe(0);
  await page.keyboard.up('ArrowRight');
  await page.locator('canvas').first().click();
  await page.keyboard.down('ArrowDown');
  await button(page, /Ván mới/).click();
  await expect.poll(async () => (await state(page, sid)).tick).toBe(0);
  await page.keyboard.press('Space');
  expect((await step(page, sid)).actions[0]).toBe(5);
  expect((await step(page, sid)).actions[0]).toBe(0);
  await page.keyboard.up('ArrowDown');
});

test('socket-only disconnect clears the local key registry before reconnect', async ({ page }) => {
  await page.addInitScript(() => {
    const NativeSocket = window.WebSocket;
    const sockets: WebSocket[] = [];
    (window as any).__qaSockets = sockets;
    window.WebSocket = class extends NativeSocket {
      constructor(url: string | URL, protocols?: string | string[]) { super(url, protocols); sockets.push(this); }
    };
  });
  const sid = await realGame(page, 'play', 3, 100);
  await page.locator('canvas').first().click();
  await page.keyboard.down('ArrowRight');
  await page.evaluate(() => { for (const ws of (window as any).__qaSockets) if (ws.readyState === WebSocket.OPEN) ws.close(4000, 'QA socket drop'); });
  await expect(page.locator('body')).toContainText(/Mất kết nối|Đang kết nối lại/);
  await expect(page.locator('body')).toContainText('Đã kết nối');
  await page.keyboard.press('Space');
  expect((await step(page, sid)).actions[0]).toBe(5);
  expect((await step(page, sid)).actions[0]).toBe(0);
  await page.keyboard.up('ArrowRight');
});
