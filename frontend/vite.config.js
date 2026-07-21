import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    host: "0.0.0.0",
    port: 5173,
    open: true,

    proxy: {
      "/chat": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/lead": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/feedback": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/version": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/model": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/session": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/history": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/pipeline": { target: "http://127.0.0.1:8000", changeOrigin: true }
    }
  },

  build: {
    outDir: "dist",
    sourcemap: false
  }
});