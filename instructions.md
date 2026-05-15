# Albanian OCR Normalizer — Setup Instructions (Windows)

## System Requirements

Before starting, install the following:

1. **Python 3.10+**
   - Download from https://www.python.org/downloads/
   - During installation, check **"Add Python to PATH"**

2. **Node.js 18+**
   - Download from https://nodejs.org/
   - The installer includes npm automatically

3. **Tesseract OCR**
   - Download the Windows installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - During installation, note the install path (default: `C:\Program Files\Tesseract-OCR`)
   - After installing, add Tesseract to your system PATH:
     1. Open **Start Menu** and search for "Environment Variables"
     2. Click **"Edit the system environment variables"**
     3. Click **"Environment Variables"**
     4. Under **System variables**, find `Path` and click **Edit**
     5. Click **New** and add: `C:\Program Files\Tesseract-OCR`
     6. Click OK on all dialogs
     7. Restart any open terminals

---

## Backend Setup

Open **Command Prompt** or **PowerShell** and run:

```cmd
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

You should see `(.venv)` at the start of your terminal line once the virtual environment is active.

On first run, the following will be **auto-downloaded** (~1–2 GB total):
- **PaddleOCR** model weights (Albanian OCR)
- **mT5-small** model weights from Hugging Face (`google/mt5-small`)

This may take 5–10 minutes depending on your internet speed.

Start the backend server:

```cmd
uvicorn main:app --reload
```

The backend runs on `http://localhost:8000` — keep this terminal open.

---

## Frontend Setup

Open a **second** Command Prompt or PowerShell window and run:

```cmd
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173` — keep this terminal open too.

---

## Running the App

Once both servers are running, open your browser and go to:

```
http://localhost:5173
```

---

## Notes

- **No API keys required** — everything runs locally (PaddleOCR + mT5)
- **SQLite database** (`history.db`) is created automatically on first run — no database setup needed
- **GPU is optional** — if you have an NVIDIA GPU with CUDA, it will be used automatically; otherwise it falls back to CPU
- The `backend/uploads/` folder is created automatically when you first upload a file
- Every time you want to run the app, you need to activate the virtual environment first (`.venv\Scripts\activate`) before starting the backend
