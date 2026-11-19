//! AudioGhost desktop.
//!
//! A launcher window sets up a private Python environment on first run (Python,
//! PyTorch, FFmpeg and SAM-Audio, via uv), then starts the AudioGhost backend in
//! desktop mode (no Redis, no separate worker) and shows the UI it serves. The
//! backend lives in a Windows job object, so it never outlives the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State, Url};
use tauri_plugin_opener::OpenerExt;
use tauri_plugin_updater::{Update, UpdaterExt};

/// How long the backend may take to answer /health (first start imports a lot).
const START_TIMEOUT: Duration = Duration::from_secs(120);
const CHECK_IMPORTS: &str = "import torch, sam_audio, fastapi, celery; \
    print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())";

#[derive(Default, Clone, Serialize, Deserialize)]
struct Config {
    python: Option<String>,
    ffmpeg_bin: Option<String>,
}

#[cfg(windows)]
struct KillOnClose(#[allow(dead_code)] win32job::Job);
// The job is only a handle that is closed when the app exits.
#[cfg(windows)]
unsafe impl Send for KillOnClose {}
#[cfg(windows)]
unsafe impl Sync for KillOnClose {}

#[derive(Default)]
struct AppState {
    backend: Mutex<Option<Child>>,
    setup: Mutex<Option<Child>>,
    url: Mutex<Option<String>>,
    launcher: Mutex<Option<Url>>,
    /// A newer version found by `check_update`, waiting for `install_update`.
    update: Mutex<Option<Update>>,
    stopping: AtomicBool,
    #[cfg(windows)]
    job: Mutex<Option<KillOnClose>>,
}

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// Paths without the `\\?\` prefix, which PowerShell and some tools don't accept.
fn plain(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => path,
    }
}

struct Paths {
    root: PathBuf,
    resources: PathBuf,
}

impl Paths {
    fn new(app: &AppHandle) -> Result<Self, String> {
        Ok(Self {
            // The app's own data folder, not %LOCALAPPDATA%\AudioGhost: that is where the
            // per-user installer puts the program, and it must stay separate from the data.
            root: plain(app.path().app_local_data_dir().map_err(err)?),
            resources: plain(app.path().resource_dir().map_err(err)?),
        })
    }
    fn config(&self) -> PathBuf {
        self.root.join("config.json")
    }
    fn logs(&self) -> PathBuf {
        self.root.join("logs")
    }
    fn data(&self) -> PathBuf {
        self.root.join("data")
    }
    fn env_marker(&self) -> PathBuf {
        self.root.join("env").join("audioghost-env.json")
    }
    fn own_ffmpeg(&self) -> PathBuf {
        self.root.join("ffmpeg").join("bin")
    }
    fn backend(&self) -> PathBuf {
        self.resources.join("backend")
    }
    fn web(&self) -> PathBuf {
        self.resources.join("web")
    }
    fn setup_script(&self) -> PathBuf {
        self.resources.join("setup").join("setup.ps1")
    }
    fn uv(&self) -> PathBuf {
        self.resources.join("bin").join("uv.exe")
    }
}

fn load_config(paths: &Paths) -> Config {
    fs::read_to_string(paths.config())
        .ok()
        .and_then(|text| serde_json::from_str(text.trim_start_matches('\u{feff}')).ok())
        .unwrap_or_default()
}

fn save_config(paths: &Paths, config: &Config) -> Result<(), String> {
    fs::create_dir_all(&paths.root).map_err(err)?;
    let text = serde_json::to_string_pretty(config).map_err(err)?;
    fs::write(paths.config(), text).map_err(err)
}

/// A command that opens no console window.
fn command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

/// Put a child in the app's job: Windows ends it (and what it started) with the app.
fn adopt(state: &AppState, child: &Child) {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        if let Some(job) = state.job.lock().unwrap().as_ref() {
            let _ = job.0.assign_process(child.as_raw_handle() as isize);
        }
    }
    #[cfg(not(windows))]
    let _ = (state, child);
}

fn kill_tree(child: &mut Child) {
    #[cfg(windows)]
    {
        let _ = command("taskkill").args(["/T", "/F", "/PID", &child.id().to_string()]).output();
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn env_root(python: &Path) -> PathBuf {
    let dir = python.parent().unwrap_or(Path::new("."));
    let in_scripts = dir.file_name().map(|n| n.to_string_lossy().eq_ignore_ascii_case("Scripts")).unwrap_or(false);
    if in_scripts { dir.parent().unwrap_or(dir).to_path_buf() } else { dir.to_path_buf() }
}

/// PATH for a Python environment: FFmpeg first, then the env's own folders
/// (a venv's Scripts, a conda env's Library\bin), then the user's PATH.
fn search_path(python: &Path, ffmpeg: Option<&Path>) -> std::ffi::OsString {
    let root = env_root(python);
    let mut parts: Vec<PathBuf> = Vec::new();
    if let Some(dir) = ffmpeg {
        parts.push(dir.to_path_buf());
    }
    parts.extend([root.clone(), root.join("Library").join("bin"), root.join("Scripts")]);
    if let Some(existing) = std::env::var_os("PATH") {
        parts.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(parts).unwrap_or_default()
}

fn tail(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

// --------------------------------------------------------------------------- status

#[derive(Serialize)]
struct Gpu {
    name: String,
    memory_gb: f64,
}

#[derive(Serialize)]
struct Status {
    ready: bool,
    python: Option<String>,
    root: String,
    free_gb: f64,
    gpu: Option<Gpu>,
    existing: Vec<String>,
    version: String,
}

fn detect_gpu() -> Option<Gpu> {
    let out = command("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.lines().next()?.split(',').map(str::trim);
    let name = parts.next()?.to_string();
    let megabytes: f64 = parts.next()?.parse().ok()?;
    Some(Gpu { name, memory_gb: (megabytes / 1024.0).round() })
}

fn free_gb(path: &Path) -> f64 {
    let mut probe = path.to_path_buf();
    while !probe.exists() {
        if !probe.pop() {
            break;
        }
    }
    fs2::available_space(&probe).map(|bytes| (bytes as f64 / 1_073_741_824.0 * 10.0).round() / 10.0).unwrap_or(0.0)
}

/// Conda environments made by AudioGhost's install.bat, for people who already have one.
fn existing_envs() -> Vec<String> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(home) = std::env::var("USERPROFILE") {
        for name in ["miniconda3", "anaconda3", "miniforge3", "mambaforge"] {
            roots.push(PathBuf::from(&home).join(name));
        }
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        roots.push(PathBuf::from(local).join("miniconda3"));
    }
    if let Ok(conda) = std::env::var("CONDA_EXE") {
        if let Some(root) = Path::new(&conda).parent().and_then(Path::parent) {
            roots.push(root.to_path_buf());
        }
    }
    for drive in ["C", "D", "E"] {
        for name in ["miniconda", "miniconda3", "anaconda3", r"ProgramData\miniconda3"] {
            roots.push(PathBuf::from(format!(r"{drive}:\{name}")));
        }
    }
    let mut found: Vec<String> = Vec::new();
    for root in roots {
        let python = root.join("envs").join("audioghost").join("python.exe");
        let text = python.to_string_lossy().to_string();
        if python.is_file() && !found.iter().any(|f| f.eq_ignore_ascii_case(&text)) {
            found.push(text);
        }
    }
    found
}

#[tauri::command]
async fn status(app: AppHandle) -> Result<Status, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let paths = Paths::new(&app)?;
        let config = load_config(&paths);
        let ready = config.python.as_deref().map(|p| Path::new(p).is_file()).unwrap_or(false);
        Ok(Status {
            ready,
            python: config.python,
            root: paths.root.to_string_lossy().to_string(),
            free_gb: free_gb(&paths.root),
            gpu: detect_gpu(),
            existing: existing_envs(),
            version: app.package_info().version.to_string(),
        })
    })
    .await
    .map_err(err)?
}

// --------------------------------------------------------------------------- setup

#[derive(Clone, Serialize)]
struct SetupLine {
    text: String,
}

#[derive(Clone, Serialize)]
struct SetupDone {
    ok: bool,
    message: String,
}

#[derive(Deserialize)]
struct Marker {
    python: String,
    ffmpeg_bin: Option<String>,
}

fn read_marker(paths: &Paths) -> Result<Config, String> {
    let text = fs::read_to_string(paths.env_marker())
        .map_err(|_| "The installer finished, but its environment marker is missing.".to_string())?;
    let marker: Marker = serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(err)?;
    Ok(Config { python: Some(marker.python), ffmpeg_bin: marker.ffmpeg_bin })
}

#[tauri::command]
fn install(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let paths = Paths::new(&app)?;
    let mut setup = state.setup.lock().unwrap();
    if setup.is_some() {
        return Err("Setup is already running.".into());
    }
    fs::create_dir_all(paths.logs()).map_err(err)?;
    let log = Arc::new(Mutex::new(File::create(paths.logs().join("setup.log")).map_err(err)?));
    let mut child = command("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"])
        .arg(paths.setup_script())
        .arg("-Root")
        .arg(&paths.root)
        .arg("-Uv")
        .arg(paths.uv())
        .arg("-Backend")
        .arg(paths.backend())
        .env("NO_COLOR", "1")
        .env("UV_NO_PROGRESS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Could not start the installer: {e}"))?;
    adopt(&state, &child);
    let streams: [Box<dyn Read + Send>; 2] = [Box::new(child.stdout.take().unwrap()), Box::new(child.stderr.take().unwrap())];
    *setup = Some(child);
    drop(setup);

    for stream in streams {
        let (app, log) = (app.clone(), log.clone());
        thread::spawn(move || {
            for chunk in BufReader::new(stream).split(b'\n').map_while(Result::ok) {
                let text = String::from_utf8_lossy(&chunk).trim_end_matches('\r').to_string();
                let _ = writeln!(log.lock().unwrap(), "{text}");
                let _ = app.emit("setup-line", SetupLine { text });
            }
        });
    }

    let app = app.clone();
    thread::spawn(move || {
        let finished = loop {
            thread::sleep(Duration::from_millis(300));
            let state = app.state::<AppState>();
            let mut setup = state.setup.lock().unwrap();
            let Some(child) = setup.as_mut() else { break None }; // cancelled
            match child.try_wait() {
                Ok(Some(status)) => {
                    *setup = None;
                    break Some(status.success());
                }
                Ok(None) => {}
                Err(_) => {
                    *setup = None;
                    break Some(false);
                }
            }
        };
        let done = match finished {
            None => SetupDone { ok: false, message: "Setup was cancelled.".into() },
            Some(false) => SetupDone { ok: false, message: String::new() },
            Some(true) => match read_marker(&paths).and_then(|config| save_config(&paths, &config)) {
                Ok(()) => SetupDone { ok: true, message: String::new() },
                Err(message) => SetupDone { ok: false, message },
            },
        };
        let _ = app.emit("setup-done", done);
    });
    Ok(())
}

#[tauri::command]
fn cancel_install(state: State<'_, AppState>) {
    if let Some(mut child) = state.setup.lock().unwrap().take() {
        kill_tree(&mut child);
    }
}

#[tauri::command]
async fn use_python(app: AppHandle, python: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let paths = Paths::new(&app)?;
        let python = PathBuf::from(python.trim().trim_matches('"'));
        if !python.is_file() {
            return Err("That file doesn't exist.".into());
        }
        let ffmpeg = Some(paths.own_ffmpeg()).filter(|dir| dir.join("avcodec-62.dll").is_file());
        let out = command(&python)
            .args(["-c", CHECK_IMPORTS])
            .env("PATH", search_path(&python, ffmpeg.as_deref()))
            .env("PYTHONUTF8", "1")
            .output()
            .map_err(err)?;
        if !out.status.success() {
            return Err(format!("This Python can't run AudioGhost:\n{}", tail(&String::from_utf8_lossy(&out.stderr), 8)));
        }
        let config = Config {
            python: Some(python.to_string_lossy().to_string()),
            ffmpeg_bin: ffmpeg.map(|dir| dir.to_string_lossy().to_string()),
        };
        save_config(&paths, &config)?;
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    })
    .await
    .map_err(err)?
}

// --------------------------------------------------------------------------- backend

fn free_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(err)?;
    Ok(listener.local_addr().map_err(err)?.port())
}

fn healthy(port: u16) -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(400)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    if stream.write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").is_err() {
        return false;
    }
    let mut head = [0u8; 32];
    let read = stream.read(&mut head).unwrap_or(0);
    let head = String::from_utf8_lossy(&head[..read]);
    head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
}

/// The backend's exit code once it has exited (and forget it), else None.
fn exited(state: &AppState) -> Option<i32> {
    let mut backend = state.backend.lock().unwrap();
    let status = backend.as_mut()?.try_wait().ok()??;
    *backend = None;
    Some(status.code().unwrap_or(-1))
}

fn stop_backend(state: &AppState) {
    state.stopping.store(true, Ordering::SeqCst);
    if let Some(mut child) = state.backend.lock().unwrap().take() {
        kill_tree(&mut child);
    }
    *state.url.lock().unwrap() = None;
}

fn navigate(app: &AppHandle, url: &str) {
    if let (Some(window), Ok(url)) = (app.get_webview_window("main"), Url::parse(url)) {
        let _ = window.navigate(url);
    }
}

/// If the backend stops on its own, go back to the launcher and say so.
fn watch_backend(app: AppHandle) {
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(800));
        let state = app.state::<AppState>();
        if let Some(code) = exited(&state) {
            *state.url.lock().unwrap() = None;
            if !state.stopping.load(Ordering::SeqCst) {
                if let Some(mut launcher) = state.launcher.lock().unwrap().clone() {
                    launcher.set_query(Some(&format!("exited={code}")));
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.navigate(launcher);
                    }
                }
            }
            return;
        }
        if state.backend.lock().unwrap().is_none() {
            return;
        }
    });
}

fn start_backend(app: &AppHandle) -> Result<String, String> {
    let state = app.state::<AppState>();
    let running = state.url.lock().unwrap().clone();
    if let Some(url) = running {
        if state.backend.lock().unwrap().is_some() {
            navigate(app, &url);
            return Ok(url);
        }
    }
    let paths = Paths::new(app)?;
    let config = load_config(&paths);
    let python = PathBuf::from(config.python.clone().ok_or("AudioGhost isn't set up yet.")?);
    let ffmpeg = config.ffmpeg_bin.as_ref().map(PathBuf::from);
    let port = free_port()?;
    fs::create_dir_all(paths.logs()).map_err(err)?;
    fs::create_dir_all(paths.data()).map_err(err)?;
    let log = File::create(paths.logs().join("backend.log")).map_err(err)?;
    let child = command(&python)
        .args(["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", &port.to_string()])
        .current_dir(paths.backend())
        .env("PATH", search_path(&python, ffmpeg.as_deref()))
        .env("AUDIOGHOST_INPROCESS", "1")
        .env("AUDIOGHOST_STATIC_DIR", paths.web())
        .env("AUDIOGHOST_DATA_DIR", paths.data())
        .env("PYTHONUTF8", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().map_err(err)?))
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|e| format!("Could not start AudioGhost: {e}"))?;
    adopt(&state, &child);
    state.stopping.store(false, Ordering::SeqCst);
    *state.backend.lock().unwrap() = Some(child);

    let started = Instant::now();
    while !healthy(port) {
        if let Some(code) = exited(&state) {
            return Err(format!("AudioGhost stopped while starting (exit code {code})."));
        }
        if started.elapsed() > START_TIMEOUT {
            stop_backend(&state);
            return Err("AudioGhost didn't start within two minutes.".into());
        }
        thread::sleep(Duration::from_millis(300));
    }
    let url = format!("http://127.0.0.1:{port}/");
    *state.url.lock().unwrap() = Some(url.clone());
    watch_backend(app.clone());
    navigate(app, &url);
    Ok(url)
}

#[tauri::command]
async fn start(app: AppHandle) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || start_backend(&app)).await.map_err(err)?
}

// --------------------------------------------------------------------------- updates

#[derive(Serialize)]
struct UpdateInfo {
    version: String,
    notes: String,
}

#[derive(Clone, Serialize)]
struct UpdateProgress {
    downloaded: u64,
    total: Option<u64>,
}

/// Ask the release feed whether a newer version is out. The feed is the latest GitHub
/// release's latest.json (tauri.conf.json); AUDIOGHOST_UPDATE_URL replaces it for testing.
#[tauri::command]
async fn check_update(app: AppHandle, state: State<'_, AppState>) -> Result<Option<UpdateInfo>, String> {
    let handle = app.clone();
    let mut builder = app
        .updater_builder()
        .timeout(Duration::from_secs(20))
        // The installer replaces the files the backend runs from: stop it first.
        .on_before_exit(move || stop_backend(&handle.state::<AppState>()));
    if let Ok(url) = std::env::var("AUDIOGHOST_UPDATE_URL") {
        builder = builder.endpoints(vec![Url::parse(&url).map_err(err)?]).map_err(err)?;
    }
    let update = builder.build().map_err(err)?.check().await.map_err(err)?;
    let info = update.as_ref().map(|u| UpdateInfo { version: u.version.clone(), notes: u.body.clone().unwrap_or_default() });
    *state.update.lock().unwrap() = update;
    Ok(info)
}

/// Download the update, check its signature and run its installer, which closes the app and
/// opens the new version.
#[tauri::command]
async fn install_update(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let update = state.update.lock().unwrap().take().ok_or("There is no update to install.")?;
    let mut downloaded: u64 = 0;
    let bytes = update
        .download(
            move |chunk, total| {
                downloaded += chunk as u64;
                let _ = app.emit("update-progress", UpdateProgress { downloaded, total });
            },
            || {},
        )
        .await
        .map_err(err)?;
    update.install(bytes).map_err(err)
}

// --------------------------------------------------------------------------- folders

#[tauri::command]
fn open_folder(app: AppHandle, which: String) -> Result<(), String> {
    let paths = Paths::new(&app)?;
    let target = match which.as_str() {
        "logs" => paths.logs(),
        "app" => paths.root.clone(),
        "setup-log" => paths.logs().join("setup.log"),
        "backend-log" => paths.logs().join("backend.log"),
        "notices" => paths.resources.join("THIRD_PARTY_NOTICES.txt"),
        _ => return Err(format!("Unknown folder: {which}")),
    };
    if target.extension().is_none() {
        fs::create_dir_all(&target).map_err(err)?;
    } else if !target.is_file() {
        return Err("There is no log yet.".into());
    }
    app.opener().open_path(target.to_string_lossy(), None::<&str>).map_err(err)
}

/// Forget the chosen Python, so the launcher offers setup again.
#[tauri::command]
fn reset(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    stop_backend(&state);
    let paths = Paths::new(&app)?;
    let _ = fs::remove_file(paths.config());
    Ok(())
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(AppState::default())
        .setup(|app| {
            let state = app.state::<AppState>();
            #[cfg(windows)]
            if let Ok(job) = win32job::Job::create() {
                if let Ok(mut info) = job.query_extended_limit_info() {
                    info.limit_kill_on_job_close();
                    let _ = job.set_extended_limit_info(&mut info);
                }
                *state.job.lock().unwrap() = Some(KillOnClose(job));
            }
            if let Some(window) = app.get_webview_window("main") {
                *state.launcher.lock().unwrap() = window.url().ok();
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            status,
            install,
            cancel_install,
            use_python,
            start,
            check_update,
            install_update,
            open_folder,
            reset
        ])
        .build(tauri::generate_context!())
        .expect("error while building AudioGhost");

    app.run(|app, event| {
        if let RunEvent::Exit = event {
            let state = app.state::<AppState>();
            stop_backend(&state);
            let setup = state.setup.lock().unwrap().take();
            if let Some(mut child) = setup {
                kill_tree(&mut child);
            }
        }
    });
}
