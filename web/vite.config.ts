import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// В разработке UI ходит в API через прокси: origin один, CORS не нужен (ragglet-28 §12).
const api = process.env.RAGGLET_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/v1": { target: api, changeOrigin: false } },
  },
  // Зависимости ленивых страниц — заранее: иначе dev-сервер пересобирает их на лету, и в странице
  // оказываются две копии React.
  optimizeDeps: {
    include: [
      "@radix-ui/react-dialog",
      "@radix-ui/react-dropdown-menu",
      "@radix-ui/react-tooltip",
      "@codemirror/commands",
      "@codemirror/lang-yaml",
      "@codemirror/language",
      "@codemirror/lint",
      "@codemirror/search",
      "@codemirror/state",
      "@codemirror/view",
      "yaml",
      "zustand",
    ],
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 900,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["src/test-setup.ts"],
    css: { modules: { classNameStrategy: "non-scoped" } },
  },
});
