from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import uvicorn
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Import existing project components. Try package-relative imports first
# (works when running `uvicorn src.api:app`) and fall back to plain imports
# so `python src/api.py` still works during local testing.
try:
    # when module is loaded as a package (src.api)
    from .data_pipeline import LogDataPipeline
    from .model_inference import LogClassifier
    from .alert_engine import AlertEngine
    from .nlp_interface_enhanced import NLPLLMInterface
except Exception:
    # fallback when running the file directly (not as a package)
    from data_pipeline import LogDataPipeline
    from model_inference import LogClassifier
    from alert_engine import AlertEngine
    try:
        from nlp_interface_enhanced import NLPLLMInterface
    except Exception:
        NLPLLMInterface = None

# Initialize NLP interface at startup
NLP_INTERFACE = None
try:
    NLP_INTERFACE = NLPLLMInterface(model_name='t5-small')
    logger.info("NLP interface initialized successfully.")
except Exception as e:
    logger.warning(f"Could not initialize NLP interface: {e}")

app = FastAPI(
    title="Cybersecurity Log Analyzer API",
    version="1.0.0",
    description="AI-powered log analysis with NLP query interface"
)

# Allow CORS from frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
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
            # Save uploaded file content to sample.log
            sample_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample.log')
            with open(sample_log_path, 'wb') as f:
                f.write(content)
            logger.info(f"Saved uploaded log to {sample_log_path}")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.log', mode='w', encoding='utf-8') as tmp:
                tmp.write(log_text)
                tmp_path = tmp.name
            # Save log text to sample.log
            sample_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample.log')
            with open(sample_log_path, 'w', encoding='utf-8') as f:
                f.write(log_text)
            logger.info(f"Saved log text to {sample_log_path}")

        # Run pipeline
        pipeline = LogDataPipeline()
        df = pipeline.process_log_file(tmp_path)
        
        # Ensure required columns exist
        if 'message' not in df.columns:
            df['message'] = df.get('raw', '')
        if len(df) == 0:
            return {'error': 'No valid log lines found', 'rows_preview': [], 'alerts': [], 'rows_count': 0, 'timestamp': pd.Timestamp.now().isoformat()}

        # Classification
        classifier = LogClassifier(model_type=model)
        try:
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
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            classified = pd.DataFrame([{'classification': 'unknown', 'confidence': 0.0}] * len(df))

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
            'rows_count': len(df),
            'timestamp': pd.Timestamp.now().isoformat()
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def get_logs_and_alerts_from_sample():
    """Read sample.log and generate alerts for NLP context.
    Returns (DataFrame, alerts_list) or (None, None) if sample.log doesn't exist.
    """
    sample_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample.log')
    
    if not os.path.exists(sample_log_path):
        return None, None
    
    try:
        # Parse logs from sample.log
        pipeline = LogDataPipeline()
        df = pipeline.process_log_file(sample_log_path)
        
        # Generate alerts
        alert_engine = AlertEngine()
        alerts = alert_engine.generate_alerts(df)
        
        return df, alerts
    except Exception as e:
        logger.error(f"Error reading sample.log: {e}")
        return None, None


@app.post("/query")
async def query_logs(query: str = Form(...)):
    """Natural language query against logs and alerts.
    Uses the latest uploaded logs from sample.log.
    
    Args:
        query: Natural language question about logs/alerts.
        
    Returns:
        JSON with natural language response and metadata.
    """
    if NLP_INTERFACE is None or not NLP_INTERFACE.enabled:
        return {
            "error": "NLP interface not available. Ensure LangChain and transformers are installed.",
            "response": None
        }
    
    try:
        # Get real logs and alerts from sample.log
        df, alerts = get_logs_and_alerts_from_sample()
        
        # Fallback to stub data if sample.log doesn't exist
        if df is None or alerts is None:
            logger.info("sample.log not found or empty, using stub data")
            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-10-11', periods=8),
                'event': ['reboot', 'reboot', 'reboot', 'reboot', 'failed login', 'failed login', 'accepted login', 'other'],
                'user': [None, None, None, None, 'admin', 'admin', 'user1', None],
                'ip': [None, None, None, None, '192.168.0.23', '192.168.0.23', '192.168.0.10', None],
                'message': [
                    'Booting Linux',
                    'Booting Linux',
                    'Booting Linux',
                    'Booting Linux',
                    'Failed password for admin from 192.168.0.23',
                    'Failed password for admin from 192.168.0.23',
                    'Accepted password for user1',
                    'Kernel message'
                ]
            })
            alerts = [
                {
                    'timestamp': '2025-10-11T03:06:20',
                    'alert_type': 'Frequent Reboots',
                    'severity': 'Medium',
                    'message': '4 system restarts within 10 minutes — potential instability or attack.'
                }
            ]
        
        response = NLP_INTERFACE.query(query, df, alerts)
        
        return {
            "query": query,
            "response": response,
            "timestamp": pd.Timestamp.now().isoformat(),
            "nlp_model": NLP_INTERFACE.model_name
        }
    except Exception as e:
        logger.error(f"Error in NLP query: {e}")
        return {
            "error": str(e),
            "response": None
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "nlp_available": NLP_INTERFACE is not None and NLP_INTERFACE.enabled
    }


if __name__ == '__main__':
    # Run with: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host='0.0.0.0', port=8000)
