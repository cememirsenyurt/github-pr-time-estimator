// frontend/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // any request to /predict will be forwarded to localhost:5000
      "/predict": {
        target: "http://127.0.0.1:5000/",
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
