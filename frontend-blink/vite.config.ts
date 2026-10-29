/// <reference types="vitest" />
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
  ],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: [],
  },
  server: {
    port: 5173,
    // Proxy /api requests to the FastAPI backend during development.
    // This avoids CORS issues and lets VITE_API_URL default to '/api/v1'.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Heatmap images served by FastAPI's StaticFiles mount
      '/heatmaps': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
