// The launcher: first-run setup, then start the backend (the app swaps this
// window over to the UI the backend serves).
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const NEED_GB = 10;
const $ = (id) => document.getElementById(id);
let retry = () => boot();
let logFor = "backend-log";

function show(view) {
  document.querySelectorAll(".view").forEach((el) => { el.hidden = el.id !== `view-${view}`; });
}

function problem(title, text, { log = "backend-log", canReset = false, again = () => boot() } = {}) {
  $("problem-title").textContent = title;
  $("problem-text").textContent = text;
  $("reset").hidden = !canReset;
  logFor = log;
  retry = again;
  show("problem");
}

async function boot() {
  const exitCode = new URLSearchParams(location.search).get("exited");
  if (exitCode !== null) {
    history.replaceState(null, "", location.pathname);
    problem("AudioGhost stopped", `It quit unexpectedly (exit code ${exitCode}). The log says why.`,
      { again: () => start(), canReset: true });
    return;
  }
  show("checking");
  let status;
  try {
    status = await invoke("status");
  } catch (err) {
    problem("Couldn't check this PC", String(err));
    return;
  }
  $("version").textContent = `Version ${status.version}`;
  if (status.ready) {
    start();
  } else {
    setup(status);
  }
}

function setup(status) {
  const gpu = $("fact-gpu");
  if (status.gpu) {
    gpu.textContent = `${status.gpu.name} · ${status.gpu.memory_gb} GB`;
    gpu.className = "ok";
  } else {
    gpu.textContent = "No NVIDIA GPU found. AudioGhost will run on the CPU, which is very slow.";
    gpu.className = "warn";
  }
  $("fact-root").textContent = status.root;
  const space = $("fact-space");
  const enough = status.free_gb >= NEED_GB;
  space.textContent = enough ? `${status.free_gb} GB free` : `${status.free_gb} GB free, about ${NEED_GB} GB needed`;
  space.className = enough ? "ok" : "warn";
  $("install").disabled = !enough;
  $("install-hint").textContent = enough ? "About 6 GB" : "Free up some space first";

  const list = $("existing-list");
  list.replaceChildren();
  status.existing.forEach((python) => {
    const item = document.createElement("li");
    const path = document.createElement("code");
    path.textContent = python;
    const use = document.createElement("button");
    use.type = "button";
    use.textContent = "Use this";
    const result = document.createElement("p");
    result.className = "error result";
    result.hidden = true;
    use.addEventListener("click", () => useExisting(python, use, result));
    item.append(path, use, result);
    list.append(item);
  });
  $("existing").hidden = status.existing.length === 0;
  show("setup");
}

async function useExisting(python, button, result) {
  button.disabled = true;
  const label = button.textContent;
  button.textContent = "Checking…";
  result.hidden = true;
  try {
    await invoke("use_python", { python });
    start();
  } catch (err) {
    result.textContent = String(err);
    result.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

async function install() {
  const steps = [...document.querySelectorAll("#steps li")];
  steps.forEach((li) => { li.className = ""; });
  $("log").textContent = "";
  $("now").textContent = "";
  $("cancel").disabled = false;
  show("installing");
  let current = 0;
  let lastError = "";
  const lines = [];

  const offLine = await listen("setup-line", ({ payload }) => {
    const text = payload.text;
    const step = /^::step (\d+)\/(\d+) (.*)$/.exec(text);
    if (step) {
      current = Number(step[1]);
      steps.forEach((li, i) => { li.className = i + 1 < current ? "done" : i + 1 === current ? "active" : ""; });
      $("now").textContent = step[3];
      return;
    }
    if (text.startsWith("::error ")) { lastError = text.slice(8); return; }
    if (text === "::done") { steps.forEach((li) => { li.className = "done"; }); return; }
    if (!text.trim()) return;
    lines.push(text);
    if (lines.length > 400) lines.shift();
    $("log").textContent = lines.join("\n");
    $("log").scrollTop = $("log").scrollHeight;
    $("now").textContent = text;
  });
  const offDone = await listen("setup-done", ({ payload }) => {
    offLine();
    offDone();
    if (payload.ok) {
      start();
      return;
    }
    if (current > 0) steps[current - 1].className = "failed";
    const message = payload.message || lastError || "The installer stopped. The log says why.";
    problem("Setup didn't finish", message, { log: "setup-log", again: () => install() });
  });

  try {
    await invoke("install");
  } catch (err) {
    offLine();
    offDone();
    problem("Setup couldn't start", String(err), { log: "setup-log", again: () => install() });
  }
}

async function start() {
  show("starting");
  $("starting-text").textContent = "Loading the app…";
  const slow = setTimeout(() => { $("starting-text").textContent = "Still loading. The first start takes a little longer…"; }, 8000);
  try {
    await invoke("start"); // on success the window switches to the app
  } catch (err) {
    problem("AudioGhost didn't start", `${err} The log says why.`, { again: () => start(), canReset: true });
  } finally {
    clearTimeout(slow);
  }
}

async function openFolder(which) {
  try { await invoke("open_folder", { which }); } catch (err) { console.warn(err); }
}

$("install").addEventListener("click", install);
$("cancel").addEventListener("click", async () => {
  $("cancel").disabled = true;
  await invoke("cancel_install");
});
$("other-use").addEventListener("click", () => {
  const python = $("other-python").value.trim();
  if (python) useExisting(python, $("other-use"), $("other-error"));
});
$("retry").addEventListener("click", () => retry());
$("open-log").addEventListener("click", () => openFolder(logFor));
$("reset").addEventListener("click", async () => { await invoke("reset"); boot(); });
$("open-app-folder").addEventListener("click", () => openFolder("app"));
$("open-logs").addEventListener("click", () => openFolder("logs"));
boot();
