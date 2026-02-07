import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

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
      // Proxy API requests to Docker backend during development
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/api/ws': {
        target: 'ws://localhost:8001',
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
