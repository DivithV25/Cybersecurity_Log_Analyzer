# Cybersecurity Log Analyzer - File Explanations & Architecture

## 📁 Backend Files


### 1. **`data_pipeline.py`** - Log Processing & Feature Extraction
Parses raw log files, extracts structured features, and generates embeddings using DistilBERT transformer.

**Key Methods**:
- `parse_log_line()`: Regex-based log parsing
- `extract_features()`: Extracts IPs, users, events, commands
- `log_to_embedding()`: Converts log text to 768-dim vectors

**Output Example**:
```python
{
  'timestamp': datetime(2025, 10, 11, 10, 32, 21),
  'host': 'server1',
  'message': 'Failed password for admin from 192.168.0.23',
  'ip': '192.168.0.23',
  'user': 'admin',
  'event': 'failed login',
  'embedding': [0.123, -0.456, ...]  # 768-dim vector
}
```

---

### 2. **`model_inference.py`** - AI Classification Engine
Classifies log entries as "normal" or "suspicious" using Transformer or DNN models.

**Key Methods**:
- `classify_transformer()`: Uses DistilBERT directly on text
- `classify_dnn()`: Uses MLPClassifier on embeddings
- `classify()`: Routes to appropriate classifier

**Output**: Binary classification (normal/suspicious) with confidence scores

---

### 3. **`alert_engine.py`** - Pattern Detection & Alert Generation
Detects security patterns and generates alerts.

**Detection Methods**:
1. **Brute Force** (v2.0): ≥5 failed logins within 2 minutes → **Counts actual attempts**
2. **Privilege Escalation**: Detects sudo/su commands, root access
3. **Suspicious Access Time**: Logins outside work hours (7 AM - 8 PM)
4. **Frequent Reboots**: >3 reboots within 10 minutes

**Key Methods**:
- `detect_brute_force()`: Sliding window analysis (counts actual attempts, not threshold)
- `detect_privilege_escalation()`: Pattern matching for privilege changes
- `detect_suspicious_access_time()`: Time-based anomaly detection
- `detect_frequent_reboots()`: Frequency analysis

**Alert Structure**:
```json
{
  "timestamp": "2025-10-11T10:32:27",
  "alert_type": "Brute Force",
  "user": "admin",
  "ip": "192.168.0.23",
  "count": 7,
  "severity": "High",
  "message": "7 failed logins detected from IP 192.168.0.23 for user admin — possible brute-force attack."
}
```

---

### 4. **`nlp_interface_enhanced.py`** - Natural Language Query Interface (v2.0)
Enables natural language queries with smart routing and direct extraction.

**Smart Query Routing**:
- **Basic Queries** (instant, <100ms): Direct extraction for ports, IPs, users, counts
- **Complex Queries** (2-3s): Uses T5 model for analysis

**Key Methods**:
- `query()`: Routes query and returns answer
- `_is_basic_query()`: Classifies query type
- `_extract_direct_answer()`: Extracts direct answers with **brute force count extraction** using regex `(\d+)\s+failed\s+login`
- `_clean_response()`: Removes raw log data
- `_extract_ports_from_logs()`: Regex-based port extraction

**Query Examples**:
```
Basic (Direct):
- "Which ports are attacked?" → Extract ports from messages
- "How many brute force attacks?" → Extract attempt count from alerts
- "What IPs are suspicious?" → Enumerate unique IPs

Complex (T5 Model):
- "Describe the attack sequence"
- "Analyze the security timeline"
- "What pattern indicates privilege escalation?"
```

---

### 5. **`app.py`** - CLI Entry Point
Command-line interface orchestrating the entire pipeline.

**Usage**:
```bash
python src/app.py --input sample.log --model transformer
python src/app.py --input sample.log --query "How many brute force attacks?"
```

---

### 6. **`api.py`** - REST API Server (FastAPI)
Web API endpoint for frontend integration.

**Endpoints**:
- `POST /analyze`: Process logs → return alerts
- `POST /query`: Natural language query on logs
- `GET /health`: System status

**Usage**:
```bash
uvicorn src.api:app --reload --port 8000
```

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

## Quick Start Guide

### Prerequisites
- Python 3.8+
- Node.js 14+
- Git

### Installation & Setup

**1. Backend Setup**:
```bash
# Navigate to project directory
cd d:\GenAiProject\Cybersecurity_Log_Analyzer

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**2. Frontend Setup**:
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Return to root
cd ..
```

### Running the System

**Option 1: CLI Mode**
```bash
# Analyze logs from command line
python src/app.py --input sample.log --model transformer

# With NLP query
python src/app.py --input sample.log --query "How many brute force attacks?"
```

**Option 2: API + Frontend Mode**
```bash
# Terminal 1: Start backend API
uvicorn src.api:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
# Frontend available at: http://localhost:3000

---

## 📊 System Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                   USER INTERFACE                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Next.js React Frontend (Port 3000)              │  │
│  │  - File upload / Paste logs                      │  │
│  │  - Display alerts with severity                  │  │
│  │  - Query results (tabbed interface)              │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTP/POST
┌────────────────────▼─────────────────────────────────────┐
│              REST API LAYER (FastAPI)                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │  /analyze  → Process logs → Generate alerts     │   │
│  │  /query    → NLP query on logs                   │   │
│  │  /health   → System status                       │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│              PROCESSING PIPELINE                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  data_pipeline.py                               │   │
│  │  - Parse logs (regex)                           │   │
│  │  - Extract features (IP, user, event)           │   │
│  │  - Generate embeddings (DistilBERT)             │   │
│  └──────────────────────────────────────────────────┘   │
│           ↓                                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  model_inference.py                             │   │
│  │  - Classify: Transformer or DNN                 │   │
│  │  - Output: Normal/Suspicious labels             │   │
│  └──────────────────────────────────────────────────┘   │
│           ↓                                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  alert_engine.py (v2.0)                         │   │
│  │  - Detect patterns (rules)                      │   │
│  │  - Count actual attempts ⭐                     │   │
│  │  - Generate alerts (JSON)                       │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│              NLP QUERY LAYER (v2.0)                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │  nlp_interface_enhanced.py                       │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │ Query Router                              │ │   │
│  │  │ - Basic (factual) → Direct extraction    │ │   │
│  │  │ - Complex (analysis) → T5 model          │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │ Direct Extraction                         │ │   │
│  │  │ - Port/IP/User enumeration                │ │   │
│  │  │ - Attempt count extraction ⭐            │ │   │
│  │  │ - Severity counting                      │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │ T5 Model Query                            │ │   │
│  │  │ - Format prompt                           │ │   │
│  │  │ - Generate response                       │ │   │
│  │  │ - Clean output                            │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│              OUTPUT                                       │
│  - Alerts (JSON) with actual counts                      │
│  - Classified logs                                       │
│  - NLP responses (instant or AI-generated)               │
└─────────────────────────────────────────────────────────┘
```

---

## 🎓 Final Notes

This **Cybersecurity Log Analyzer** project successfully combines:

1. **Deep Learning**: Transformers for semantic understanding
2. **Machine Learning**: DNN for efficient classification
3. **Rule-Based Logic**: Temporal pattern detection
4. **NLP**: T5 model for natural language analysis
5. **Web Technology**: FastAPI + React for full-stack deployment

**Key Innovation**: Hybrid approach balancing AI intelligence with deterministic rule systems.
