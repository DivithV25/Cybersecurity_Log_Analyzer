# Cybersecurity Log Analyzer - Complete Project Explanation

## 📋 Project Overview

This is an **AI-powered Cybersecurity Log Analyzer MVP** that processes system log files to detect security threats and anomalies. The system uses a combination of **Deep Learning (Transformers)**, **Natural Language Processing (NLP)** to identify suspicious activities like brute-force attacks, privilege escalations, and unusual access patterns.

---

## 🏗️ Architecture & Workflow

### High-Level Data Flow

```
┌─────────────┐
│  Log File   │ (Input: .log files)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  data_pipeline.py   │ → Parses, cleans, extracts features, generates embeddings
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ model_inference.py  │ → Classifies logs as "normal" or "suspicious" using AI
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  alert_engine.py    │ → Detects patterns, generates security alerts
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   alerts.json       │ (Output: Security alerts)
└─────────────────────┘

Optional:
┌─────────────────────┐
│  nlp_interface.py   │ → Natural language query interface (LLM-powered)
└─────────────────────┘
```

### Component Interaction

1. **Frontend (Next.js)** → User uploads/pastes logs
2. **API (FastAPI)** → Receives logs, orchestrates processing
3. **Data Pipeline** → Parses and preprocesses logs
4. **Model Inference** → AI classification
5. **Alert Engine** → Pattern detection and alert generation
6. **NLP Interface** → Optional natural language queries

---

## 📁 File-by-File Breakdown

### Backend Files

#### 1. **`app.py`** - Main Entry Point (CLI)
**Purpose**: Command-line interface for running the analyzer

**What it does**:
- Accepts command-line arguments (`--input`, `--query`, `--model`)
- Orchestrates the entire pipeline:
  1. Calls `LogDataPipeline` to parse logs
  2. Calls `LogClassifier` to classify entries
  3. Calls `AlertEngine` to generate alerts
  4. Optionally uses `NLPLLMInterface` for natural language queries
- Displays results in the terminal

**Key Functions**:
- `main()`: Entry point that coordinates all components

**Usage**:
```bash
python src/app.py --input sample.log --model transformer
```

---

#### 2. **`api.py`** - REST API Server
**Purpose**: Web API endpoint for frontend integration

**What it does**:
- Creates a FastAPI server with CORS enabled
- Exposes `/analyze` endpoint that:
  - Accepts log files or raw text via POST
  - Runs the same pipeline as `app.py`
  - Returns JSON with parsed logs and alerts
- Handles file uploads and temporary file management

**Key Components**:
- `FastAPI` app with CORS middleware
- `/analyze` endpoint: Main analysis endpoint
- Returns structured JSON: `{rows_preview, alerts, rows_count}`

**Usage**:
```bash
uvicorn src.api:app --reload --port 8000
```

---

#### 3. **`data_pipeline.py`** - Log Processing & Feature Extraction
**Purpose**: Parses raw log files and extracts structured features

**What it does**:
1. **Parses log lines** using regex patterns
2. **Extracts structured data**: timestamp, host, process, message
3. **Feature extraction**:
   - IP addresses
   - Users (from login attempts, sudo commands)
   - Event types (failed login, accepted login, reboot, etc.)
   - Commands (from sudo/su operations)
4. **Generates embeddings** using DistilBERT transformer model

**Key Classes & Methods**:
- `LogDataPipeline`: Main pipeline class
  - `parse_log_line()`: Regex-based log parsing
  - `extract_features()`: Extracts IPs, users, events, commands
  - `log_to_embedding()`: Converts log text to vector embeddings
  - `process_log_file()`: Main processing function

**AI/ML Concepts Used**:
- **Text Embeddings**: Uses DistilBERT (`distilbert-base-uncased`) to convert log messages into numerical vectors (768-dimensional)
- **Feature Engineering**: Extracts structured features from unstructured text

**Example Output**:
```python
{
  'timestamp': datetime(2025, 10, 11, 10, 32, 21),
  'host': 'server1',
  'process': 'sshd[1234]',
  'message': 'Failed password for invalid user admin from 192.168.0.23',
  'ip': '192.168.0.23',
  'user': 'admin',
  'event': 'failed login',
  'embedding': numpy.array([0.123, -0.456, ...])  # 768-dim vector
}
```

---

#### 4. **`model_inference.py`** - AI Classification Engine
**Purpose**: Classifies log entries as "normal" or "suspicious" using AI models

**What it does**:
- Supports two model types:
  1. **Transformer Model** (default): Uses pre-trained DistilBERT fine-tuned for sentiment/classification
  2. **DNN Model**: Multi-layer Perceptron (MLP) classifier on embeddings
- Processes logs in batches for efficiency
- Returns classification labels and confidence scores

**Key Classes & Methods**:
- `LogClassifier`: Main classifier class
  - `classify_transformer()`: Uses transformer model directly on text
  - `classify_dnn()`: Uses DNN on pre-computed embeddings
  - `classify()`: Main method that routes to appropriate classifier

**AI/ML Concepts Used**:
1. **Transformer Models**:
   - Uses `distilbert-base-uncased-finetuned-sst-2-english`
   - Pre-trained on sentiment analysis, adapted for security classification
   - Input: Raw log text → Tokenization → Transformer → Softmax → Classification
   - Output: Binary classification (normal/suspicious) with confidence score

2. **Deep Neural Networks (DNN)**:
   - Multi-layer Perceptron (MLPClassifier from scikit-learn)
   - Takes embeddings as input (768-dim vectors)
   - Trained to classify embeddings into normal/suspicious

3. **Softmax Activation**: Converts raw model outputs to probability scores

**How Classification Works**:
```python
# Transformer path:
log_text → Tokenizer → Transformer Model → Logits → Softmax → [P(normal), P(suspicious)]
# If P(suspicious) > 0.5 → Label = "suspicious", else "normal"

# DNN path:
embedding (768-dim) → MLP → Probability scores → Classification
```

---

#### 5. **`alert_engine.py`** - Pattern Detection & Alert Generation
**Purpose**: Detects security patterns and generates alerts

**What it does**:
- pattern detection:
  1. **Brute Force Detection**: ≥5 failed logins from same IP/user within 2 minutes
  2. **Privilege Escalation**: Detects sudo/su commands, root access attempts
  3. **Suspicious Access Time**: Logins outside work hours (7 AM - 8 PM)
  4. **Frequent Reboots**: >3 reboots within 10 minutes
- Generates structured alerts with severity levels

**Key Classes & Methods**:
- `AlertEngine`: Main alert generation class
  - `detect_brute_force()`: Sliding window analysis for failed logins
  - `detect_privilege_escalation()`: Pattern matching for privilege changes
  - `detect_suspicious_access_time()`: Time-based anomaly detection
  - `detect_frequent_reboots()`: Frequency analysis for reboots
  - `generate_alerts()`: Orchestrates all detection methods

**AI/ML Concepts Used**:
- **Time-Series Analysis**: Sliding window algorithms for temporal patterns
- **Anomaly Detection**: Rule-based heuristics for unusual behaviors
- **Pattern Recognition**: Regex and keyword matching for security events

**Alert Structure**:
```json
{
  "timestamp": "2025-10-11T10:32:27",
  "alert_type": "Brute Force",
  "user": "admin",
  "ip": "192.168.0.23",
  "count": 5,
  "severity": "High",
  "message": "5 failed logins detected from IP 192.168.0.23 for user admin — possible brute-force attack."
}
```

---

#### 6. **`nlp_interface.py`** - Natural Language Query Interface
**Purpose**: Enables natural language queries about logs and alerts using LLMs

**What it does**:
- Uses LangChain and Hugging Face transformers
- Allows users to ask questions like:
  - "Show all suspicious activities from 192.168.0.23"
  - "Summarize last night's failed logins"
- Processes logs and alerts context, generates natural language responses

**Key Classes & Methods**:
- `NLPLLMInterface`: LLM query interface
  - `query()`: Takes natural language query, returns AI-generated answer

**AI/ML Concepts Used**:
1. **Large Language Models (LLMs)**:
   - Uses Hugging Face pipeline for text generation
   - LangChain for prompt engineering and chain construction

2. **Prompt Engineering**:
   - Structured prompts with context (logs + alerts)
   - Template-based query processing

3. **Text-to-Text Generation**: Converts user queries + context → answers

**How it Works**:
```
User Query + Logs DataFrame + Alerts List
    ↓
Prompt Template (LangChain)
    ↓
Hugging Face LLM Pipeline
    ↓
Natural Language Response
```

---

### Frontend Files

#### 7. **`frontend/pages/index.js`** - Web UI
**Purpose**: React-based user interface for log analysis

**What it does**:
- Provides a web form to:
  - Paste log text directly
  - Upload log files
- Displays results:
  - Alerts with severity indicators
  - Parsed log rows preview
- Communicates with backend API via HTTP POST

**Key Features**:
- File upload handling
- Loading states
- Error handling
- Responsive UI with styled components

---

## 🤖 AI/ML Concepts Used & How They're Applied

### 1. **Transformer Models (Deep Learning)**

**What**: Pre-trained neural networks that understand context in text

**How Used**:
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Location**: `data_pipeline.py` (embeddings) and `model_inference.py` (classification)
- **Purpose**:
  - **Embeddings**: Converts log messages into 768-dimensional vectors that capture semantic meaning
  - **Classification**: Directly classifies log text as normal/suspicious

**Technical Details**:
- Uses **self-attention mechanisms** to understand relationships between words
- **Fine-tuned** on sentiment analysis, adapted for security classification
- **Tokenization**: Converts text → token IDs → embeddings
- **Softmax**: Converts logits → probability scores

**Example**:
```python
# Input: "Failed password for invalid user admin from 192.168.0.23"
# → Tokenizer → [101, 4567, 2341, ...] (token IDs)
# → Transformer → [0.2, 0.8] (probabilities: [normal, suspicious])
# → Classification: "suspicious" (confidence: 0.8)
```

---

### 2. **Text Embeddings (Vector Representations)**

**What**: Numerical representations of text that capture semantic meaning

**How Used**:
- **Model**: DistilBERT base model
- **Location**: `data_pipeline.py` → `log_to_embedding()`
- **Purpose**: Converts unstructured log text into fixed-size vectors (768 dimensions)

**Why Important**:
- Enables similarity calculations between log entries
- Can be used for clustering, similarity search, or as input to other ML models
- Preserves semantic meaning in numerical form

**Technical Details**:
- Uses the `[CLS]` token embedding (first token) as the sentence representation
- Output: NumPy array of shape `(768,)`

---

### 3. **Deep Neural Networks (DNN)**

**What**: Multi-layer perceptron for classification

**How Used**:
- **Model**: `MLPClassifier` from scikit-learn
- **Location**: `model_inference.py` → `classify_dnn()`
- **Purpose**: Alternative classification method using embeddings as input

**Architecture**:
- Input: 768-dimensional embeddings
- Hidden layers: Multiple fully connected layers
- Output: Binary classification (normal/suspicious)

**Advantages**:
- Faster inference than transformers
- Can be trained on domain-specific data
- Lower memory footprint

---

### 4. **Natural Language Processing (NLP)**

**What**: Processing and understanding human language

**How Used**:
- **Location**: `nlp_interface.py`
- **Technologies**: LangChain, Hugging Face Transformers
- **Purpose**: Enables natural language queries about logs

**Components**:
- **Prompt Engineering**: Structured templates for LLM queries
- **Chain Construction**: LangChain orchestrates the query → context → response flow
- **Text Generation**: LLM generates human-readable answers

**Example Flow**:
```
User: "Show suspicious activities from 192.168.0.23"
    ↓
LLM receives: Query + Logs context + Alerts context
    ↓
LLM generates: "Found 5 failed login attempts from 192.168.0.23..."
```

---

### 5. **Anomaly Detection (Rule-Based + ML Hybrid)**

**What**: Identifying unusual patterns that deviate from normal behavior

**How Used**:
- **Location**: `alert_engine.py`
- **Approach**: Hybrid (Rule-based + ML classification)

**Methods**:
1. **ML-Based**: Transformer/DNN classifies individual log entries
2. **Rule-Based**: Pattern matching for:
   - Temporal patterns (brute force windows)
   - Frequency analysis (reboot counts)
   - Behavioral patterns (privilege escalation)

**Example**:
- ML detects: "This log entry looks suspicious" (confidence: 0.85)
- Rule-based detects: "5 failed logins in 2 minutes = brute force pattern"
- Combined: High-confidence alert generated

---

### 6. **Feature Engineering**

**What**: Extracting meaningful features from raw data

**How Used**:
- **Location**: `data_pipeline.py` → `extract_features()`
- **Purpose**: Converts unstructured logs into structured features

**Features Extracted**:
- **IP Addresses**: Regex pattern matching
- **Users**: Pattern matching from login messages
- **Event Types**: Keyword-based classification (failed login, reboot, etc.)
- **Commands**: Parsing sudo/su command structures
- **Timestamps**: Temporal features for time-based analysis

**Why Important**:
- Enables rule-based detection
- Provides context for ML models
- Supports filtering and aggregation

---

## 🔄 Complete Workflow Example

### Scenario: Analyzing a log file with brute force attack

1. **User Action**: Uploads `sample.log` via frontend or CLI

2. **Data Pipeline** (`data_pipeline.py`):
   ```
   Raw log: "Oct 11 10:32:21 server1 sshd[1234]: Failed password for admin from 192.168.0.23"
   ↓
   Parsed: {timestamp: 2025-10-11 10:32:21, host: server1, process: sshd[1234], message: "Failed password..."}
   ↓
   Features: {ip: "192.168.0.23", user: "admin", event: "failed login"}
   ↓
   Embedding: [0.123, -0.456, 0.789, ...] (768-dim vector)
   ```

3. **Model Inference** (`model_inference.py`):
   ```
   Log text → Transformer → [P(normal)=0.2, P(suspicious)=0.8]
   ↓
   Classification: "suspicious" (confidence: 0.8)
   ```

4. **Alert Engine** (`alert_engine.py`):
   ```
   Detects: 5 failed logins from 192.168.0.23 in 2 minutes
   ↓
   Generates Alert: {
     alert_type: "Brute Force",
     severity: "High",
     message: "5 failed logins detected..."
   }
   ```

5. **Output**: JSON file with alerts + classified log entries

---

## 🎯 Key AI/ML Innovations

1. **Hybrid Approach**: Combines deep learning (transformers) with rule-based detection
2. **Transfer Learning**: Uses pre-trained models fine-tuned for security
3. **Embedding-Based Analysis**: Converts text to vectors for semantic understanding
4. **Natural Language Interface**: LLM-powered query system for user interaction
5. **Temporal Pattern Detection**: Time-series analysis for behavioral anomalies

---

## 📊 Data Flow Summary

```
Log File (Text)
    ↓
[Parsing & Feature Extraction] → Structured DataFrame
    ↓
[Embedding Generation] → Vector Representations
    ↓
[AI Classification] → Normal/Suspicious Labels
    ↓
[Pattern Detection] → Security Alerts
    ↓
[Optional NLP Query] → Natural Language Answers
    ↓
Output: Alerts + Classified Logs
```

---

## 🛠️ Technologies & Libraries

- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained transformer models
- **LangChain**: LLM orchestration and prompt management
- **scikit-learn**: DNN classifier and utilities
- **pandas**: Data manipulation
- **FastAPI**: REST API framework
- **Next.js/React**: Frontend framework

---

This project demonstrates a practical application of modern AI/ML techniques in cybersecurity, combining the power of deep learning with domain-specific rule-based systems for comprehensive threat detection.

