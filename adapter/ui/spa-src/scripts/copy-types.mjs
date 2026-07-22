// Copia contracts/epp.gen.ts (raíz del repo) → src/types/epp.gen.ts.
//
// No es un segundo codegen: contracts/epp.gen.ts ya lo genera
// scripts/gen_epp_types.py (fuente: adapter/epp_core.py) y CI verifica que
// esté en sync (ver .github/workflows/ci.yml). Este script solo lo copia
// hacia dentro de spa-src/ para que Vite (root = spa-src/) pueda resolverlo
// como import relativo sin tocar `server.fs.allow`.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..", "..", "..", "..");
const src = join(repoRoot, "contracts", "epp.gen.ts");
const destDir = join(here, "..", "src", "types");
const dest = join(destDir, "epp.gen.ts");

mkdirSync(destDir, { recursive: true });
copyFileSync(src, dest);
console.log(`Copied ${src} -> ${dest}`);
