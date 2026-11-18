<#
  First-run installer of the AudioGhost desktop app (Windows).

  Creates a private Python environment with uv: Python 3.11, PyTorch (CUDA build
  picked from the GPU), FFmpeg 8.1 and SAM-Audio, the same versions install.bat
  installs. It needs no conda, Redis or git: git dependencies come as archives.

  The app reads these lines from stdout:
    ::step <n>/<total> <title>   a new step starts
    ::error <message>            the install failed
    ::done                       the install finished
  Everything else is log output (uv writes its own to stderr).
#>
param(
  [Parameter(Mandatory = $true)][string]$Root,
  [Parameter(Mandatory = $true)][string]$Uv,
  [Parameter(Mandatory = $true)][string]$Backend,
  [string]$Cuda = "auto"
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false

$TorchVersion = "2.9.0"
$TorchvisionVersion = "0.24.0"
$SamAudio = "https://github.com/facebookresearch/sam-audio/archive/bb4c6999d2677c7402360e426afc01ddfad6dce0.zip"
$FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip"
# SAM-Audio's git dependencies, as archives of the commits install.bat ends up with.
$Overrides = @(
  "dacvae @ https://github.com/facebookresearch/dacvae/archive/34e0b0dbb6400140da62f8f6366359e2f0377a1c.zip",
  "imagebind @ https://github.com/facebookresearch/ImageBind/archive/53680b02d7e37b19b124fa37bae4b6c98c38f5be.zip",
  "laion-clap @ https://github.com/lematt1991/CLAP/archive/b85cb7ecac0af7d064cbf03ebd990e709e959fe7.zip",
  "perception-models @ https://github.com/facebookresearch/perception_models/archive/e72b6810b1133e1c879f2cc965d276eb73803f1f.zip",
  "pytorchvideo @ https://github.com/facebookresearch/pytorchvideo/archive/6cdc929315aab1b5674b6dcf73b16ec99147735f.zip"
)
$NeedGb = 10
$Total = 7

function Step([int]$n, [string]$title) { Write-Output "::step $n/$Total $title" }
function Fail([string]$message) { Write-Output "::error $message"; exit 1 }
function Check([string]$message) { if ($LASTEXITCODE -ne 0) { Fail $message } }

New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Cache = Join-Path $Root "cache"
New-Item -ItemType Directory -Force -Path $Cache | Out-Null
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "python"
$env:UV_CACHE_DIR = Join-Path $Cache "uv"
$env:UV_LINK_MODE = "copy"
$EnvDir = Join-Path $Root "env"
$Python = Join-Path $EnvDir "Scripts\python.exe"
$Ffmpeg = Join-Path $Root "ffmpeg"

# --------------------------------------------------------------------------- 1
Step 1 "Checking this PC"
$drive = (Resolve-Path $Root).Path.Substring(0, 1)
$freeGb = [math]::Round((Get-PSDrive -Name $drive).Free / 1GB, 1)
Write-Output "Install folder: $Root ($freeGb GB free on ${drive}:)"
if ($freeGb -lt $NeedGb) { Fail "Not enough disk space: $freeGb GB free on ${drive}:, about $NeedGb GB needed." }
if ($Cuda -eq "auto") {
  $Cuda = "cpu"
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($smi) {
    $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1)
    if ($cap) {
      $major = [int]($cap.Trim().Split(".")[0])
      if ($major -ge 12) { $Cuda = "cu128" } else { $Cuda = "cu126" }
    }
  }
}
if ($Cuda -eq "cpu") { Write-Output "No NVIDIA GPU found: AudioGhost will run on the CPU (very slow)." }
else { Write-Output "PyTorch build: $Cuda" }

# --------------------------------------------------------------------------- 2
Step 2 "Installing Python 3.11"
& $Uv venv $EnvDir --python 3.11 --allow-existing
Check "Could not create the Python environment."

# --------------------------------------------------------------------------- 3
Step 3 "Installing PyTorch (about 2.5 GB)"
$index = "https://download.pytorch.org/whl/$Cuda"
& $Uv pip install --python $Python `
  "torch==$TorchVersion+$Cuda" "torchvision==$TorchvisionVersion+$Cuda" "torchaudio==$TorchVersion+$Cuda" `
  --index-url $index --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
Check "Could not install PyTorch."

# --------------------------------------------------------------------------- 4
Step 4 "Installing FFmpeg 8.1"
if (-not (Test-Path (Join-Path $Ffmpeg "bin\avcodec-62.dll"))) {
  $zip = Join-Path $Cache "ffmpeg.zip"
  $unpacked = Join-Path $Cache "ffmpeg_unpacked"
  & curl.exe --fail --location --retry 3 --silent --show-error --output $zip $FfmpegUrl
  Check "Could not download FFmpeg."
  if (Test-Path $unpacked) { Remove-Item $unpacked -Recurse -Force }
  Expand-Archive -Path $zip -DestinationPath $unpacked -Force
  $inner = Get-ChildItem $unpacked -Directory | Select-Object -First 1
  if (Test-Path $Ffmpeg) { Remove-Item $Ffmpeg -Recurse -Force }
  Move-Item $inner.FullName $Ffmpeg
  Remove-Item $unpacked -Recurse -Force
  Remove-Item $zip -Force
}
if (-not (Test-Path (Join-Path $Ffmpeg "bin\avcodec-62.dll"))) { Fail "FFmpeg could not be installed." }
Write-Output "FFmpeg: $Ffmpeg\bin"

# --------------------------------------------------------------------------- 5
Step 5 "Installing SAM-Audio"
$overrideFile = Join-Path $Cache "overrides.txt"
[IO.File]::WriteAllLines($overrideFile, $Overrides)
& $Uv pip install --python $Python -c (Join-Path $Backend "constraints.txt") --override $overrideFile "sam_audio @ $SamAudio"
Check "Could not install SAM-Audio."

# --------------------------------------------------------------------------- 6
Step 6 "Installing AudioGhost's packages"
& $Uv pip install --python $Python -r (Join-Path $Backend "requirements.txt") -c (Join-Path $Backend "constraints.txt")
Check "Could not install AudioGhost's packages."

# --------------------------------------------------------------------------- 7
Step 7 "Checking the installation"
$env:PATH = "$Ffmpeg\bin;$EnvDir;$EnvDir\Scripts;$env:PATH"
& $Python -c "import torch, torchcodec, sam_audio, fastapi, celery; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
Check "The installation check failed. Show the log for details."
$marker = [ordered]@{
  version = 1
  cuda = $Cuda
  python = $Python
  ffmpeg_bin = "$Ffmpeg\bin"
  created = (Get-Date).ToString("o")
}
[IO.File]::WriteAllText((Join-Path $EnvDir "audioghost-env.json"), ($marker | ConvertTo-Json), (New-Object System.Text.UTF8Encoding $false))
# The downloads are no longer needed; give the space back.
& $Uv cache clean | Out-Null
Write-Output "::done"
