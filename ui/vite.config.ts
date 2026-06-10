import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { resolveBackendPort } from './src/config/backendPort';

// Single source of truth for the dev-proxy backend port (issue #7163).
const backendPort = resolveBackendPort();

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
    port: 5180,
    strictPort: true,
    open: false,
    proxy: {
      // Proxy API requests to the local backend (8000) by default; set
      // VITE_BACKEND_PORT=8001 for the containerized topology (issue #7163).
      '/api': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true,
      },
      '/api/ws': {
        target: `ws://localhost:${backendPort}`,
        ws: true,
      },
    },
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
