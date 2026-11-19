// Third-party notices for what the installer redistributes: uv.exe, the Rust crates compiled
// into AudioGhost.exe, the JavaScript packages in the UI and its fonts. Each component is listed
// with its license, then every distinct license text follows once, with the components it covers.
// Called by prepare.mjs; also runs alone: node scripts/notices.mjs [path to uv.exe]
import { execFileSync, execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const desktop = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const frontend = join(desktop, "..", "frontend");
const licenses = join(desktop, "licenses");
const LICENSE_FILE = /^(licen[cs]e|copying|notice|unlicense)([-_.].*)?$/i;

export function writeNotices({ uv, out = join(desktop, "src-tauri", "resources", "THIRD_PARTY_NOTICES.txt") } = {}) {
  const texts = new Map(); // sha256 -> { text, users }
  const addText = (text, user) => {
    const clean = text.replace(/\r\n/g, "\n").trim();
    if (!clean) return false;
    const key = createHash("sha256").update(clean).digest("hex");
    if (!texts.has(key)) texts.set(key, { text: clean, users: [] });
    texts.get(key).users.push(user);
    return true;
  };
  const addFiles = (dir, user) => {
    if (!dir || !existsSync(dir)) return 0;
    let found = 0;
    for (const name of readdirSync(dir).sort()) {
      const path = join(dir, name);
      if (LICENSE_FILE.test(name) && statSync(path).isFile() && addText(readFileSync(path, "utf8"), user)) found += 1;
    }
    return found;
  };
  const warnings = [];
  const sections = [];

  // uv.exe and the fonts: license texts kept in desktop/licenses.
  let uvVersion = "";
  try {
    uvVersion = execFileSync(uv || "uv", ["--version"], { encoding: "utf8" }).trim().split(/\s+/)[1] || "";
  } catch {
    warnings.push("could not run uv --version");
  }
  const own = [
    ["uv", uvVersion, "MIT OR Apache-2.0", "https://github.com/astral-sh/uv", ["uv-LICENSE-MIT.txt", "uv-LICENSE-APACHE.txt"], "bin/uv.exe"],
    ["Inter", "", "OFL-1.1", "https://github.com/rsms/inter", ["Inter-OFL.txt"], "font, in the UI"],
    ["Space Grotesk", "", "OFL-1.1", "https://github.com/floriankarsten/space-grotesk", ["SpaceGrotesk-OFL.txt"], "font, in the UI"],
  ];
  sections.push(["Bundled tools and fonts", own.map(([name, version, license, url, files, where]) => {
    for (const file of files) {
      if (!existsSync(join(licenses, file)) || !addText(readFileSync(join(licenses, file), "utf8"), name)) {
        warnings.push(`missing ${file}`);
      }
    }
    return `${name}${version ? ` ${version}` : ""} (${where}) - ${license} - ${url}`;
  })]);

  // Rust crates linked into AudioGhost.exe: normal dependencies of the app, for Windows.
  try {
    const meta = JSON.parse(execFileSync("cargo", ["metadata", "--format-version", "1",
      "--filter-platform", "x86_64-pc-windows-msvc", "--manifest-path", join(desktop, "src-tauri", "Cargo.toml")],
      { encoding: "utf8", maxBuffer: 512 * 1024 * 1024 }));
    const nodes = new Map(meta.resolve.nodes.map((node) => [node.id, node]));
    const packages = new Map(meta.packages.map((pkg) => [pkg.id, pkg]));
    const used = new Set();
    const stack = [meta.resolve.root];
    while (stack.length) {
      const id = stack.pop();
      if (used.has(id)) continue;
      used.add(id);
      for (const dep of nodes.get(id)?.deps ?? []) {
        if (dep.dep_kinds.some((kind) => kind.kind === null)) stack.push(dep.pkg);
      }
    }
    used.delete(meta.resolve.root);
    const lines = [...used].map((id) => packages.get(id)).filter(Boolean)
      .sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version))
      .map((pkg) => {
        const user = `${pkg.name} ${pkg.version}`;
        const files = addFiles(dirname(pkg.manifest_path), user);
        const authors = (pkg.authors || []).join(", ");
        return `${user} - ${pkg.license || pkg.license_file || "see its repository"}` +
          (pkg.repository ? ` - ${pkg.repository}` : "") + (files ? "" : authors ? ` - by ${authors}` : "");
      });
    sections.push([`Rust crates in AudioGhost.exe (${lines.length})`, lines]);
  } catch (error) {
    warnings.push(`Rust crates: ${error.message.split("\n")[0]}`);
  }

  // JavaScript packages in the UI: the frontend's production dependency tree.
  try {
    let raw;
    try {
      raw = execSync("npm ls --omit=dev --all --long --json", { cwd: frontend, encoding: "utf8", maxBuffer: 512 * 1024 * 1024, stdio: ["ignore", "pipe", "ignore"] });
    } catch (error) {
      raw = error.stdout; // npm ls exits non-zero for problems it can still report
    }
    const found = new Map();
    const walk = (deps) => {
      for (const [name, dep] of Object.entries(deps || {})) {
        if (dep.path && dep.version && !found.has(`${name}@${dep.version}`)) {
          found.set(`${name}@${dep.version}`, { name, version: dep.version, path: dep.path, license: dep.license });
        }
        walk(dep.dependencies);
      }
    };
    walk(JSON.parse(raw).dependencies);
    const lines = [...found.values()].sort((a, b) => a.name.localeCompare(b.name)).map((pkg) => {
      const user = `${pkg.name} ${pkg.version}`;
      addFiles(pkg.path, user);
      const license = typeof pkg.license === "string" ? pkg.license : pkg.license?.type || "see its package";
      return `${user} - ${license}`;
    });
    sections.push([`JavaScript packages in the UI (${lines.length})`, lines]);
  } catch (error) {
    warnings.push(`JavaScript packages: ${error.message.split("\n")[0]}`);
  }

  const rule = (char) => char.repeat(78);
  const parts = [
    "AudioGhost desktop: third-party notices",
    rule("="),
    "",
    "AudioGhost is MIT-licensed (https://github.com/0x0funky/audioghost-ai). This installer also",
    "contains the components below. Each is listed with its license; every license text follows",
    "once, with the components it applies to.",
    "",
    "Not part of this installer: Python, PyTorch (BSD-3-Clause), FFmpeg (an LGPL build from",
    "github.com/BtbN/FFmpeg-Builds), SAM-Audio (SAM License) and the other Python packages are",
    "downloaded by the setup on first start, from their own sources and under their own licenses.",
    "The SAM-Audio model weights are downloaded on first use, under the SAM License, which the app",
    "shows before the first download.",
    "",
  ];
  for (const [title, lines] of sections) {
    parts.push(title, rule("-"), ...lines, "");
  }
  parts.push("License texts", rule("-"), "");
  let n = 0;
  for (const { text, users } of texts.values()) {
    n += 1;
    parts.push(rule("="), `[${n}] Applies to: ${users.join("; ")}`, rule("="), "", text, "");
  }
  // An incomplete notices file must not ship: fail the build instead.
  if (warnings.length) throw new Error(`[notices] ${warnings.join("; ")}`);
  writeFileSync(out, parts.join("\n") + "\n", "utf8");
  console.log(`[notices] ${out}: ${sections.map(([title]) => title).join(", ")}; ${texts.size} license texts`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  writeNotices({ uv: process.argv[2], out: process.argv[3] });
}
