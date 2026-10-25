import { expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { test, ready, button, realGame, state, step, unobscuredBoard } from './support';

test('settings are keyboard accessible and offer reduced motion', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await ready(page);
  await page.keyboard.press('Tab');
  const focused = page.locator(':focus');
  await expect(focused).toBeVisible();
  expect(await focused.evaluate(el => {
    const style = getComputedStyle(el);
    return style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0 || style.boxShadow !== 'none';
  }), 'Keyboard focus must have an observable indicator').toBeTruthy();
  await button(page, /Cài đặt|Thiết lập|Settings/i).click();
  const motion = page.getByRole('checkbox', { name: /Giảm chuyển động|Giảm hiệu ứng|Reduced motion/i }).first();
  await expect(motion).toBeVisible();
  await motion.check();
  await button(page, /Đóng|Xong|Hoàn tất|Quay lại|Close|Done/i).click();
  await button(page, /Cài đặt|Thiết lập|Settings/i).click();
  await expect(motion).toBeChecked();
});

test('mobile board is unobscured and touch direction/bomb controls work', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const sid = await realGame(page);
  await unobscuredBoard(page);
  const right = button(page, /^Phải$|Sang phải|Move right|^Right$/i);
  await expect(right).toBeVisible();
  await right.dispatchEvent('pointerdown', { pointerId: 1, pointerType: 'touch', button: 0, buttons: 1 });
  await right.dispatchEvent('pointerup', { pointerId: 1, pointerType: 'touch', button: 0 });
  expect((await step(page, sid)).actions[0]).toBe(4);
  const bomb = button(page, /^Bom$|Đặt bom|Place bomb/i);
  await expect(bomb).toBeVisible();
  await bomb.click();
  expect((await step(page, sid)).actions[0]).toBe(5);
  await page.screenshot({ path: testInfo.outputPath('gameplay-mobile.png'), fullPage: true });
});

test('four robot slots, local new assets and self-hosted font are rendered', async ({ page }, testInfo) => {
  const assets: Array<{ url: string; type: string; status: number }> = [];
  page.on('response', response => {
    const type = response.request().resourceType();
    if (['image', 'font'].includes(type)) assets.push({ url: response.url(), type, status: response.status() });
  });
  const sid = await realGame(page, 'watch', 4);
  await unobscuredBoard(page);
  expect((await state(page, sid)).agents).toHaveLength(4);
  const text = await page.locator('body').innerText();
  const numbered = (start: number) => Array.from({ length: 4 }, (_, i) => start + i).every(slot =>
    new RegExp(`(?:A|P|#|Robot\\s*|Tác tử\\s*)${slot}\\b`, 'i').test(text));
  expect(numbered(0) || numbered(1), 'All four slots need readable numeric identities; either zero or one based numbering is valid').toBeTruthy();
  await page.evaluate(() => document.fonts.ready);
  expect(assets.filter(a => a.type === 'font').length, 'Vietnamese font must be self-hosted and actually loaded').toBeGreaterThan(0);
  const appOrigin = new URL(page.url()).origin;
  for (const asset of assets) {
    expect(new URL(asset.url).origin, `Asset must be local: ${asset.url}`).toBe(appOrigin);
    expect(asset.status, `Asset should resolve: ${asset.url}`).toBeLessThan(400);
    expect(asset.url.toLowerCase()).not.toContain('emulogic');
  }
  const legacyDir = path.resolve('..', 'assets');
  const legacyHashes = new Set(fs.existsSync(legacyDir) ? fs.readdirSync(legacyDir).filter(p => p.endsWith('.png')).map(p => crypto.createHash('sha256').update(fs.readFileSync(path.join(legacyDir, p))).digest('hex')) : []);
  for (const asset of assets.filter(a => a.type === 'image')) {
    const response = await page.request.get(asset.url);
    const hash = crypto.createHash('sha256').update(await response.body()).digest('hex');
    expect(legacyHashes.has(hash), `Legacy PNG must not appear in web build: ${asset.url}`).toBeFalsy();
  }
  await page.screenshot({ path: testInfo.outputPath('four-robots-and-hud.png'), fullPage: true });
  await step(page, sid);
  await page.screenshot({ path: testInfo.outputPath('animated-frame.png'), fullPage: true });
});
