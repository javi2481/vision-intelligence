import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// SPA Fase 1 (addendum-s2-spa-s3): monta bajo /app/, build sin Node en el
// host de runtime (Docker corre `npm run build` en el stage node, ver
// adapter/Dockerfile). outDir apunta a adapter/ui/spa/, servido por FastAPI
// (StaticFiles) — ver adapter/app.py.
export default defineConfig({
  base: "/app/",
  plugins: [react()],
  build: {
    outDir: "../spa",
    emptyOutDir: true,
  },
});
