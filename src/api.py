from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import uvicorn
from typing import Optional

import pandas as pd

# Import existing project components. Try package-relative imports first
# (works when running `uvicorn src.api:app`) and fall back to plain imports
# so `python src/api.py` still works during local testing.
try:
    # when module is loaded as a package (src.api)
    from .data_pipeline import LogDataPipeline
    from .model_inference import LogClassifier
    from .alert_engine import AlertEngine
except Exception:
    # fallback when running the file directly (not as a package)
    from data_pipeline import LogDataPipeline
    from model_inference import LogClassifier
    from alert_engine import AlertEngine

app = FastAPI(title="Cybersecurity Log Analyzer API")

# Allow CORS from frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    log_text: Optional[str] = None
    model: Optional[str] = "transformer"


@app.post("/analyze")
async def analyze(log_text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), model: str = Form("transformer")):
    """Analyze logs provided either as raw text (log_text) or an uploaded file (file).
    Returns JSON with parsed rows (first 20) and alerts.
    """
    # Obtain log content
    if file is None and not log_text:
        return {"error": "No log_text or file provided"}

    # Write to a temporary file so existing pipeline (which expects a filepath) can be reused.
    tmp_path = None
    try:
        if file is not None:
            suffix = os.path.splitext(file.filename)[1] or ".log"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.log', mode='w', encoding='utf-8') as tmp:
                tmp.write(log_text)
                tmp_path = tmp.name

        # Run pipeline
        pipeline = LogDataPipeline()
        df = pipeline.process_log_file(tmp_path)

        # Classification
        classifier = LogClassifier(model_type=model)
        if model == 'transformer':
            classified = classifier.classify(messages=df['message'])
        else:
            # if embeddings available
            emb = df['embedding'].tolist()
            # convert list of arrays (or None) to numpy array; filter rows without embedding
            import numpy as np
            emb_arr = np.array([e for e in emb if e is not None])
            if len(emb_arr) == 0:
                classified = pd.DataFrame([{'classification': 'unknown', 'confidence': 0.0}] * len(df))
            else:
                classified = classifier.classify(embeddings=emb_arr)
                # If some rows lacked embeddings, pad with unknowns to match length
                if len(classified) < len(df):
                    pad_len = len(df) - len(classified)
                    pad = pd.DataFrame([{'classification': 'unknown', 'confidence': 0.0}] * pad_len)
                    classified = pd.concat([classified, pad], ignore_index=True)

        df = df.reset_index(drop=True).join(classified)

        # Alerts
        alert_engine = AlertEngine()
        alerts = alert_engine.generate_alerts(df)

        # Prepare a small preview of parsed logs to return
        preview_cols = ['timestamp', 'host', 'process', 'event', 'user', 'ip', 'message', 'classification', 'confidence']
        preview_df = df[preview_cols].head(50)
        # Convert timestamp to ISO strings
        def safe_convert(v):
            try:
                return v.isoformat() if pd.notnull(v) else None
            except Exception:
                return str(v)
        preview = []
        for _, r in preview_df.iterrows():
            row = {c: (safe_convert(r[c]) if c == 'timestamp' else (None if pd.isna(r[c]) else r[c])) for c in preview_cols}
            preview.append(row)

        return {
            'rows_preview': preview,
            'alerts': alerts,
            'rows_count': len(df)
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == '__main__':
    # Run with: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host='0.0.0.0', port=8000)
