import { expect } from '@playwright/test';
import { test, openSetup, field, choose, button, boundTraining } from './support';

test('training phase/count and episode validation are enforced in the UI', async ({ page }) => {
  await openSetup(page, 'training');
  const phase = field(page, /Giai đoạn|Pha|Phase/i);
  const agents = field(page, /Số tác tử|Số robot|Số người|Số lượng|Agents/i);
  for (const value of ['C', 'D']) {
    await choose(phase, value);
    expect(await agents.inputValue()).toBe('2');
    if (await agents.isDisabled()) {
      expect(await agents.inputValue()).toBe('2');
    } else if (await agents.evaluate(el => el.tagName === 'SELECT')) {
      const enabled = await agents.locator('option:enabled').evaluateAll(options => options.map(o => (o as HTMLOptionElement).value));
      expect(enabled.filter(v => v !== '2')).toEqual([]);
    } else await expect(agents).toBeDisabled();
  }
  await choose(phase, 'E');
  for (const count of ['2', '3', '4']) {
    await choose(agents, count);
    expect(await agents.inputValue()).toBe(count);
  }
  const episodes = field(page, /Số tập|Số ván|Số lượt|Episodes|Episode/i);
  await episodes.fill('0');
  let started = false;
  page.on('request', r => { if (r.method() === 'POST' && /\/api\/training$/.test(new URL(r.url()).pathname)) started = true; });
  const start = button(page, /Bắt đầu huấn luyện|Bắt đầu|Khởi chạy|Start training/i);
  if (await start.isEnabled()) await start.click();
  expect(started, 'Invalid episode count must not start a training worker').toBeFalsy();
  expect(await episodes.evaluate((el: HTMLInputElement) => !el.checkValidity()) || await page.locator('body').innerText().then(t => /hợp lệ|ít nhất|phải.*[1-9]|lỗi/i.test(t))).toBeTruthy();
});

test('bounded real training shows progress, actual metrics, board and controls', async ({ page }, testInfo) => {
  await boundTraining(page);
  await openSetup(page, 'training');
  await choose(field(page, /Giai đoạn|Pha|Phase/i), 'C');
  await field(page, /Số tập|Số ván|Số lượt|Episodes|Episode/i).fill('2');
  const responsePromise = page.waitForResponse(r => r.request().method() === 'POST' && /\/api\/training$/.test(new URL(r.url()).pathname));
  await button(page, /Bắt đầu huấn luyện|Bắt đầu|Khởi chạy|Start training/i).click();
  const response = await responsePromise;
  expect(response.ok()).toBeTruthy();
  const { job_id } = await response.json();
  expect(job_id).toBeTruthy();
  await expect(button(page, /Tạm dừng|Tiếp tục|Pause|Resume/i)).toBeVisible();
  await expect(button(page, /Dừng.*lưu|Dừng và lưu|Stop.*save/i)).toBeVisible();
  await expect.poll(async () => {
    const r = await page.request.get(`/api/training/${job_id}`);
    return (await r.json()).episode;
  }, { timeout: 40_000 }).toBe(2);
  await expect(page.locator('body')).toContainText(/WR|Tỷ lệ thắng/i);
  await expect(page.locator('body')).toContainText(/Active.?WR|Thắng chủ động/i);
  await expect(page.locator('body')).toContainText(/Reward|Phần thưởng/i);
  await expect(page.locator('body')).toContainText(/Loss|Hàm mất mát|Mất mát/i);
  await expect(page.locator('canvas').first()).toBeVisible();
  const job = await (await page.request.get(`/api/training/${job_id}`)).json();
  expect(job.frame.grid).toHaveLength(9);
  expect(job.frame.max_steps).toBeLessThanOrEqual(8);
  expect(job.metrics).toBeTruthy();
  await page.screenshot({ path: testInfo.outputPath('training-real-metrics.png'), fullPage: true });
});
