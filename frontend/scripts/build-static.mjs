// Static export of the UI for the desktop app: the backend serves these files
// itself (AUDIOGHOST_STATIC_DIR), so every API URL is relative (same origin).
// Output: frontend/out
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const next = require.resolve("next/dist/bin/next");
const env = { ...process.env, AUDIOGHOST_STATIC_EXPORT: "1", NEXT_PUBLIC_API_BASE: "same-origin" };
const result = spawnSync(process.execPath, [next, "build"], { stdio: "inherit", env });
process.exit(result.status ?? 1);
