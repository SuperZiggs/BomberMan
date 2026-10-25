import { expect, test as base, type Page, type Locator } from '@playwright/test';

export type State = { session_id: string; match_id: string; tick?: number; step?: number;
  paused?: boolean; running?: boolean; status?: string; done: boolean; actions: number[];
  agents: Array<{ row: number; col: number; agent_id: number }>; grid: number[][]; bombs: unknown[] };

export const currentTick = (s: State) => s.tick ?? s.step;
export const isPaused = (s: State) => s.paused === true || s.running === false || s.status === 'paused';

export const test = base.extend<{ cleanApp: void }>({
  cleanApp: [async ({ page }, use) => {
    const sessions = new Set<string>();
    const jobs = new Set<string>();
    page.on('response', async response => {
      if (response.request().method() !== 'POST' || !response.ok()) return;
      try {
        if (/\/api\/sessions$/.test(new URL(response.url()).pathname)) sessions.add((await response.json()).session_id);
        if (/\/api\/training$/.test(new URL(response.url()).pathname)) jobs.add((await response.json()).job_id);
      } catch { /* A failed test may close a page before its final response body arrives. */ }
    });
    await use();
    for (const job of jobs) await page.request.post(`/api/training/${job}/stop`, { data: {} }).catch(() => {});
    for (const sid of sessions) await page.request.delete(`/api/sessions/${sid}`).catch(() => {});
  }, { auto: true }],
});

export async function ready(page: Page) {
  let navigation = '';
  try { await page.goto('/', { waitUntil: 'domcontentloaded' }); }
  catch (e) { navigation = String(e); }
  expect(navigation, 'Runnable loopback browser application must answer GET /; start the documented server').toBe('');
  await expect(page.locator('body'), 'Frontend must render the Vietnamese game menu').toContainText(/Chơi|Đấu|Theo dõi|Huấn luyện/i);
}

export const button = (page: Page, name: RegExp): Locator => page.getByRole('button', { name }).first();
export const field = (page: Page, name: RegExp): Locator => page.getByLabel(name).first();

export async function choose(locator: Locator, value: string) {
  await expect(locator, 'Setup control must have an accessible label').toBeVisible();
  const tag = await locator.evaluate(el => el.tagName.toLowerCase());
  if (tag === 'select') await locator.selectOption(value);
  else await locator.fill(value);
}

export async function openSetup(page: Page, mode: 'play' | 'watch' | 'training') {
  await ready(page);
  const names = {
    play: /Chơi ngay|Chơi trực tiếp|Tự chơi|Người chơi|^Chơi$|^Play$/i,
    watch: /Xem AI|AI thi đấu|Theo dõi AI|Xem đấu|Quan sát|^Theo dõi$|Spectat|^Watch$/i,
    training: /Huấn luyện|Đào tạo|^Train|^Training/i,
  };
  await button(page, names[mode]).click();
}

export async function realGame(page: Page, mode: 'play' | 'watch' = 'play', count = 3, maxSteps = 8) {
  // Only bound the real engine through its documented public config; never mock a response.
  await page.route('**/api/sessions', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    const config = route.request().postDataJSON();
    await route.continue({ postData: JSON.stringify({ ...config, max_steps: maxSteps, seed: 912747 }) });
  });
  await openSetup(page, mode);
  await choose(field(page, /Số tác tử|Số robot|Số người|Số lượng|Agents/i), String(count));
  const creation = page.waitForResponse(r => /\/api\/sessions$/.test(new URL(r.url()).pathname) && r.request().method() === 'POST');
  await button(page, /Bắt đầu|Vào trận|Tạo ván|Khởi chạy|^Chơi$|Start/i).click();
  const response = await creation;
  expect(response.ok(), `Real model match creation failed: ${await response.text()}`).toBeTruthy();
  const data = await response.json();
  expect(data.session_id).toBeTruthy();
  await expect(page.locator('canvas').first()).toBeVisible();
  return data.session_id as string;
}

export async function state(page: Page, sid: string): Promise<State> {
  const response = await page.request.get(`/api/sessions/${sid}`);
  expect(response.ok()).toBeTruthy();
  return (await response.json()).snapshot;
}

export async function step(page: Page, sid: string) {
  const before = await state(page, sid);
  await button(page, /Một bước|Từng bước|Tiến một|^Bước$|Single.?step|^Step$/i).click();
  await expect.poll(async () => currentTick(await state(page, sid))).toBe((currentTick(before) || 0) + 1);
  return state(page, sid);
}

export async function boundTraining(page: Page) {
  await page.route('**/api/training', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    const config = route.request().postDataJSON();
    await route.continue({ postData: JSON.stringify({ ...config, episodes: Math.min(config.episodes, 2), max_steps: 8 }) });
  });
}

export async function unobscuredBoard(page: Page) {
  const board = page.locator('canvas').first();
  const bounds = await board.boundingBox();
  expect(bounds, 'Phaser board must occupy visible space').not.toBeNull();
  const viewport = page.viewportSize()!;
  expect(bounds!.width).toBeGreaterThan(200);
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width + 1);
  const overlap = await page.getByRole('button').evaluateAll((buttons, b) => buttons.filter(el => {
    const r = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    if (r.width === 0 || r.height === 0 || style.visibility === 'hidden' || style.display === 'none') return false;
    return r.left < b.x + b.width - 1 && r.right > b.x + 1 && r.top < b.y + b.height - 1 && r.bottom > b.y + 1;
  }).map(el => el.textContent || el.getAttribute('aria-label')), bounds!);
  expect(overlap, 'HUD/touch/menu buttons must not obscure board cells').toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
}
