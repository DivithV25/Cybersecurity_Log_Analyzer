# Frontend for Cybersecurity Log Analyzer

This is a minimal Next.js frontend that allows pasting logs or uploading a log file and sends it to the backend API at `http://localhost:8000/analyze`.

Quick start (Windows PowerShell):

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

Make sure the Python backend API is running (FastAPI/uvicorn):

```powershell
# from repository root
# create venv and install requirements first
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
```

Then use the frontend to submit logs and view alerts.
