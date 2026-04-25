import { defineConfig } from "vite";

const target = "http://127.0.0.1:7860";

export default defineConfig({
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/reset": {
        target,
        changeOrigin: true,
      },
      "/step": {
        target,
        changeOrigin: true,
      },
      "/state": {
        target,
        changeOrigin: true,
      },
      "/tasks": {
        target,
        changeOrigin: true,
      },
      "/api-root": {
        target,
        changeOrigin: true,
        rewrite: () => "/",
      },
    },
  },
});
