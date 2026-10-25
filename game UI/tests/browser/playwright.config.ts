import { defineConfig } from '@playwright/test';
import path from 'node:path';

export default defineConfig({
  testDir: '.',
  testMatch: '*.spec.ts',
  fullyParallel: false,
  workers: 1,
  timeout: 45_000,
  expect: { timeout: 8_000 },
  retries: 0,
  reporter: [['list'], ['json', { outputFile: path.resolve('tests/_artifacts/browser-results.json') }]],
  outputDir: path.resolve('tests/_artifacts/browser'),
  use: {
    baseURL: process.env.BOMBERMAN_BASE_URL || 'http://127.0.0.1:8000',
    browserName: 'chromium',
    launchOptions: {
      executablePath: process.env.BOMBERMAN_CHROME || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
    viewport: { width: 1440, height: 1000 },
    screenshot: 'only-on-failure',
    trace: process.env.BOMBERMAN_TRACE === '1' ? 'retain-on-failure' : 'off',
  },
});
