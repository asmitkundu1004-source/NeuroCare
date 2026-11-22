import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// BASE lets the same build serve from a domain root (Vercel, Netlify, the FastAPI
// server) or from a sub-path such as GitHub Pages: BASE=/repo-name/ npm run build
export default defineConfig({
  base: process.env.BASE || '/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
  optimizeDeps: { exclude: ['onnxruntime-web', '@mlc-ai/web-llm'] },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1500,
  },
})
