# AudioGhost AI 啟動指南

> 最簡單的方式：執行 `install.bat` 一次，之後每次用 `start.bat` 啟動、`stop.bat` 關閉。
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

## 4. 安裝 FFmpeg 7（TorchCodec 需要）
```powershell
conda install -c conda-forge "ffmpeg=7" -y
```

## 5. 安裝 SAM Audio（固定版本）
```powershell
pip install "git+https://github.com/facebookresearch/sam-audio.git@bb4c6999d2677c7402360e426afc01ddfad6dce0"
```

## 6. 安裝 Backend 依賴（會把 transformers / huggingface_hub 鎖在相容版本）
```powershell
cd backend
pip install -r requirements.txt
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

1. 點擊右上角「Connect HuggingFace」
2. 到 https://huggingface.co/facebook/sam-audio-large 申請存取權限（若之後使用 -tv 精準模型出現 403，也到該模型頁面申請）
3. 建立 Access Token：https://huggingface.co/settings/tokens
4. 貼上 Token 並連線
5. 第一次分離會自動下載模型（Small 5.1 GB / Base 7.7 GB / Large 14.9 GB），進度會顯示在畫面上

## 功能使用

- **上傳**：拖放、點擊或用鍵盤（Tab 到上傳區按 Enter）
- **描述聲音**：快捷按鈕或自行輸入（vocals、a dog barking、crowd noise…）
- **時間鎖定（Temporal Lock）**：在波形上選一段「目標聲音清楚出現」的區域，模型會把它當作提示（整段檔案仍會處理）
- **速度 / 品質**：Fast（8 步，約快 2 倍）/ Balanced（16 步，Meta 預設）/ High（32 步）
- **立體聲**：保留左右聲道（運算量約 2 倍）
- **精準模型（-tv）**：Meta 的 -tv checkpoint，選對目標聲音的準確度更高
- **輸出**：Isolated（目標聲音）/ Without（其他聲音），可下載 WAV / FLAC / MP3 或一次打包 ZIP
- **取消**：處理中可隨時取消（會在下一個推論步驟停止）
