// frontend/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // already had:
      "/predict": "http://localhost:8080",
      // add this:
      "/estimate": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
});
