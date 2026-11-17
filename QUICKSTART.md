# AudioGhost AI 啟動指南

> 最簡單的方式：Windows 執行 `install.bat` 一次，之後用 `start.bat` 啟動、`stop.bat` 關閉。
> macOS（Apple Silicon）/ Linux：`./install.sh`、`./start.sh`、`./stop.sh`（Mac 目前用 CPU 運算，較慢）。
> 以下是手動安裝步驟（Windows / PowerShell）。

## 1. 啟動 Redis
```powershell
# 方法 A：install.bat 下載的離線版（start.bat 會自動啟動）
# 方法 B：Docker
docker-compose up -d
```

## 2. 建立 Conda 環境
```powershell
conda create -n audioghost python=3.11 -y
conda activate audioghost
```

## 3. 安裝 PyTorch 2.9
```powershell
# RTX 20/30/40 系列（CUDA 12.6）
pip install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
# RTX 50 系列（Blackwell）請改用 cu128
pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

## 4. 安裝 FFmpeg（TorchCodec 需要）
Windows：下載 FFmpeg 8 的 shared build（例如 BtbN 的 `ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip`：https://github.com/BtbN/FFmpeg-Builds/releases/tag/latest ），解壓縮後把 `bin` 資料夾加入 PATH（`install.bat` 會自動做）。
目前請避免在 Windows 上用 conda-forge 的 FFmpeg：它的套件裝出兩份不同的 `intl-8.dll`，會導致載入失敗（#10）。

macOS / Linux：
```bash
conda install -c conda-forge "ffmpeg=7" -y
```

## 5. 安裝 SAM Audio（固定版本）
```powershell
pip install -c backend\constraints.txt "git+https://github.com/facebookresearch/sam-audio.git@bb4c6999d2677c7402360e426afc01ddfad6dce0"
```

## 6. 安裝 Backend 依賴（會把 transformers / huggingface_hub 鎖在相容版本）
```powershell
cd backend
pip install -c constraints.txt -r requirements.txt
```

## 7. 啟動 Backend API
```powershell
cd backend
uvicorn main:app --port 8000
```

## 8. 啟動 Celery Worker（新終端機）
```powershell
conda activate audioghost
cd backend
celery -A workers.celery_app worker --loglevel=info --pool=solo
```

## 9. 啟動 Frontend（新終端機）
```powershell
cd frontend
npm ci
npm run dev
```

## 10. 開啟瀏覽器
訪問 http://localhost:3000

## 首次使用

1. 在開始頁右側「This computer」卡片點「Connect」（或點右上角的狀態標籤）
2. 到 https://huggingface.co/facebook/sam-audio-large 申請存取權限（若之後使用 -tv 精準模型出現 403，也到該模型頁面申請）
3. 建立 Access Token：https://huggingface.co/settings/tokens
4. 貼上 Token 並連線
5. 第一次分離會自動下載模型（Small 5.1 GB / Base 7.7 GB / Large 14.9 GB），進度會顯示在該軌道上

## 功能使用（v1.2）

- **上傳**：拖放或點「Choose a file」。每個檔案是一個工作階段，會列在開始頁的「Recent」，重新整理頁面也不會遺失（保留 24 小時）
- **描述聲音**：在右側面板輸入（按 Enter 加入）或點建議，例如 speech、vocals、background music、a dog barking。一次可以放好幾個，每個聲音各自一軌
- **Everything else**：模型依序把每個軌道的聲音拿掉後剩下的聲音（接力：每個新的聲音都從前面剩下的再拿掉一次），全部都是模型輸出，聽起來仍像原本的錄音
- **聽到的是哪個版本**：全部軌道都在 0 dB 時播放原音；只調整一軌時播放模型自己的「沒有這個聲音」版本加上該軌；調整多軌時播放 Everything else 加上各軌。匯出也依同樣規則。刪除已完成的軌道會把它的聲音加回 Everything else
- **混音器**：每軌有 S（獨奏）、M（靜音）和音量；按住「Hold to hear the original」可以對照原音；空白鍵播放／暫停
- **時間鎖定**：先在面板點選一個新的聲音，再在時間軸上拖曳：「Sound is here」標出聲音清楚出現的地方，「Not here」標出沒有它的地方（模型當作提示，整段仍會處理）
- **模型 / 品質**：Small / Base / Large；Fast（8 步）/ Balanced（16 步，Meta 預設）/ High（32 步）
- **進階**：立體聲（立體聲檔案預設開啟；關掉約快 2 倍但會變成單聲道）、精準模型（-tv）、完整 FP32、最長分段長度
- **匯出**：「What you hear」輸出目前的混音；也可以把各軌打包成 ZIP，或（影片）輸出帶有混音的影片。格式：24-bit WAV / FLAC / MP3
- **取消**：軌道上的 ✕ 可以取消處理中的軌道，或移除排隊中的軌道
