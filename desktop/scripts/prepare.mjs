// Gathers what the app bundles, in src-tauri/resources:
//   backend/   the Python backend (without tests, outputs, uploads or a saved token)
//   web/       the UI as a static export (frontend: npm run build:static)
//   setup/     the first-run installer
//   bin/uv.exe the Python package manager the installer uses
// --skip-web (or AUDIOGHOST_SKIP_WEB=1) reuses an existing web/ build.
import { spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, rmSync, statSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const desktop = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(desktop, "..");
const out = join(desktop, "src-tauri", "resources");
const skipWeb = process.argv.includes("--skip-web") || process.env.AUDIOGHOST_SKIP_WEB === "1";

// An allow-list, not a deny-list: the backend folder also holds runtime data
// (sessions, uploads, model checkpoints, a saved token) that must never ship.
const SKIP_DIRS = new Set(["tests", "outputs", "uploads", "__pycache__"]);
const KEEP_FILES = new Set(["requirements.txt", "constraints.txt"]);
const keepBackend = (source) => {
  const name = basename(source);
  if (statSync(source).isDirectory()) return !name.startsWith(".") && !SKIP_DIRS.has(name);
  return name.endsWith(".py") || KEEP_FILES.has(name);
};

for (const part of ["backend", "setup", "bin"]) rmSync(join(out, part), { recursive: true, force: true });
mkdirSync(out, { recursive: true });

cpSync(join(repo, "backend"), join(out, "backend"), { recursive: true, filter: keepBackend });

if (!skipWeb || !existsSync(join(out, "web", "index.html"))) {
  const frontend = join(repo, "frontend");
  const build = spawnSync(process.execPath, [join(frontend, "scripts", "build-static.mjs")], { cwd: frontend, stdio: "inherit" });
  if (build.status !== 0) process.exit(build.status ?? 1);
  rmSync(join(out, "web"), { recursive: true, force: true });
  cpSync(join(frontend, "out"), join(out, "web"), { recursive: true });
}

cpSync(join(desktop, "setup"), join(out, "setup"), { recursive: true });

function findUv() {
  if (process.env.AUDIOGHOST_UV) return process.env.AUDIOGHOST_UV;
  const where = spawnSync(process.platform === "win32" ? "where" : "which", ["uv"], { encoding: "utf8" });
  const first = (where.stdout || "").split(/\r?\n/).find((line) => line.trim());
  if (!first) throw new Error("uv not found: install it (https://docs.astral.sh/uv/) or set AUDIOGHOST_UV");
  return first.trim();
}
mkdirSync(join(out, "bin"), { recursive: true });
cpSync(findUv(), join(out, "bin", process.platform === "win32" ? "uv.exe" : "uv"));
console.log(`[prepare] resources ready in ${out}`);
