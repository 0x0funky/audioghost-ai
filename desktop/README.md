# AudioGhost desktop (Windows)

A Tauri 2 (Rust) app around the same backend and UI as the web version. The user never
needs a terminal, conda, Redis or git.

## How it works

1. **First start.** The launcher window (`ui/`) offers two choices:
   - *Install* runs `setup/setup.ps1`, which uses the bundled `uv` to create a private
     environment: Python 3.11, PyTorch 2.9 (the CUDA build is picked from the GPU), FFmpeg 8.1,
     SAM-Audio and AudioGhost's packages. These are the same versions `install.bat` installs.
     About 3 GB is downloaded and 6 GB installed. Git dependencies are fetched as archives.
   - *Use this* reuses a conda environment made by `install.bat`, after checking that it can
     import torch and SAM-Audio.
2. **Every start.** The app starts the backend from the chosen Python:
   ```
   python -m uvicorn main:app --port <free port>
   ```
   It runs in desktop mode (`AUDIOGHOST_INPROCESS=1`: jobs run in the backend process, with no
   Redis or worker). The backend serves the UI itself (`AUDIOGHOST_STATIC_DIR`). Once `/health`
   answers, the window switches to it.
3. **Lifetime.**
   - The backend runs inside a Windows job object, so it ends with the app, even if the app
     crashes.
   - If the backend stops on its own, the window goes back to the launcher and offers the log.

Models are not bundled. They come from the Hugging Face cache, so models downloaded by the web
version (or by any other tool, `HF_HOME` included) are reused. A token saved with `hf auth login`
is picked up automatically.

## Where things live

`%LOCALAPPDATA%\ai.audioghost.desktop\` (the program itself installs to `%LOCALAPPDATA%\AudioGhost\`):

| Folder | What |
|---|---|
| `config.json` | The Python the app starts (written after setup, or by *Use this*) |
| `env\`, `python\` | The private environment and the Python it was made from |
| `ffmpeg\` | FFmpeg 8.1 shared build |
| `data\` | Sessions, uploads, results and the saved Hugging Face token |
| `logs\` | `setup.log` and `backend.log` |

The uninstaller can delete this folder too (tick *Delete the application data*).

## Build

Needs Rust (stable, MSVC), Node 20+ and `uv` on PATH. The frontend's `node_modules` must be installed.

```bash
cd desktop
npm install
npm run build        # release app + NSIS installer in src-tauri/target/release/bundle/nsis/
npm run dev          # debug build with a live window
```

`scripts/prepare.mjs` runs first, from `beforeBuildCommand`. It gathers what the app bundles into
`src-tauri/resources/`:
- the backend (without tests, outputs or a saved token);
- the static UI export (`frontend: npm run build:static`);
- the setup script;
- `uv.exe`.

Set `AUDIOGHOST_SKIP_WEB=1` to reuse the last UI export.

To drive the window from tests, start the app with
`WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9340` and attach a DevTools client.

## Not done yet

- **Code signing:** Windows SmartScreen warns about the unsigned installer.
- **Auto-update.**
- **macOS:** the shell builds there, but setup is Windows-only for now, and Macs run SAM-Audio on the CPU.
