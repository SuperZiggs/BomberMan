import { defineConfig } from 'vite';

export default defineConfig({
  server: { proxy: { '/api': { target: 'http://127.0.0.1:8000', ws: true } } },
  build: { chunkSizeWarningLimit: 1600, rollupOptions: { output: { manualChunks: (id: string) => id.includes('node_modules/phaser') ? 'phaser' : undefined } } },
});
