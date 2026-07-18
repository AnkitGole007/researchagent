import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the frontend runs on :5173 and proxies /api to the FastAPI backend
// on :8000, so the app can use same-origin relative URLs ("/api/...") exactly
// like it will in production (where FastAPI serves the built frontend).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
