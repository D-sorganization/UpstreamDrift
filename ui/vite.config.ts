import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {
  DEV_SERVER_PORT,
  buildDevProxy,
  resolveApiPort,
} from './src/config/devProxy';

// Port and proxy-key ordering live in `src/config/devProxy.ts` so they can be
// regression-tested without loading esbuild. See issues #8076 and #8077.
const API_PORT = resolveApiPort(process.env.VITE_API_PORT);

export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      // hls.js ESM entry is missing in v1.6.x — alias to the CJS dist
      'hls.js': path.resolve(__dirname, 'node_modules/hls.js/dist/hls.js'),
    },
  },

  server: {
    port: DEV_SERVER_PORT,
    strictPort: true,
    open: false,
    // Proxy API requests to the local Python API during development.
    // Override the port with VITE_API_PORT when running the API elsewhere.
    proxy: buildDevProxy(API_PORT),
  },

  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          'three': ['three', '@react-three/fiber', '@react-three/drei'],
          'react': ['react', 'react-dom'],
          'charts': ['recharts'],
        },
      },
    },
  },
});
