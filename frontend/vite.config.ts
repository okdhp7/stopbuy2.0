import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { defineConfig } from "vite";

const rootDir = import.meta.dirname;
const backendHttpUrl = process.env.VITE_BACKEND_HTTP_URL || "http://127.0.0.1:8010";
const backendWsProxyTarget = process.env.VITE_BACKEND_WS_PROXY_TARGET || "ws://127.0.0.1:8010";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(rootDir, "client", "src"),
      "@shared": path.resolve(rootDir, "shared"),
    },
  },
  root: path.resolve(rootDir, "client"),
  build: {
    outDir: path.resolve(rootDir, "dist"),
    emptyOutDir: true,
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/ws": {
        target: backendWsProxyTarget,
        ws: true,
        changeOrigin: true,
      },
      "/health": {
        target: backendHttpUrl,
        changeOrigin: true,
      },
    },
  },
});
