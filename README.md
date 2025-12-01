# Cybersecurity Log Analyzer (MVP)

## Overview
An **AI-powered Cybersecurity Log Analyzer** that reads, processes, and analyzes system log files to detect unusual or potentially harmful activities, triggering intelligent alerts using Machine Learning and Generative AI.

---

## Features
- Accepts raw log files as input (e.g., `.log` or text-based system logs)
- Parses, cleans, and normalizes log data
- Uses deep learning (transformer-based) models to classify log entries as *normal* or *suspicious*
- Monitors event frequency and generates alerts for repeated failed logins, unusual access times, privilege escalations, and more
- CLI interface for viewing anomalies, alerts, and summaries
- Supports natural language queries (via LLM, optional)

---

## Architecture

```
log file → [data_pipeline.py] → [model_inference.py] → [alert_engine.py] → alerts.json
															 ↓
												[nlp_interface.py] (optional)
```

- **data_pipeline.py**: Parses and preprocesses logs
- **model_inference.py**: Embeds and classifies log entries
- **alert_engine.py**: Detects patterns and generates alerts
- **nlp_interface.py**: Natural language query interface (optional)
- **app.py**: Entry point

---

## Setup & Installation

1. **Clone the repo**
2. **Install dependencies**:

	```bash
	pip install torch transformers pandas langchain
	```

3. **Run the analyzer**:

	```bash
	python src/app.py --input sample.log
	.\.venv\Scripts\python.exe src\app.py --input sample.log 
	```

	Alerts will be saved to `alerts.json` by default.

---

## Example Usage

**Input Log Snippet:**
```
Oct 11 10:32:21 server1 sshd[1234]: Failed password for invalid user admin from 192.168.0.23
Oct 11 10:32:23 server1 sshd[1234]: Failed password for invalid user admin from 192.168.0.23
Oct 11 10:32:24 server1 sshd[1234]: Failed password for invalid user admin from 192.168.0.23
Oct 11 10:32:25 server1 sshd[1234]: Failed password for invalid user admin from 192.168.0.23
Oct 11 10:32:27 server1 sshd[1234]: Failed password for invalid user admin from 192.168.0.23
```

**Sample Output:**
```
[DONE] 1 alerts generated. Saved to alerts.json

Sample alerts:
{'timestamp': '2025-10-11 10:32:27', 'alert_type': 'Brute Force', 'user': 'admin', 'ip': '192.168.0.23', 'count': 5, 'severity': 'High', 'message': '5 failed logins detected from IP 192.168.0.23 for user admin — possible brute-force attack.'}
```

---

## Natural Language Query (Optional)

Enable the LLM interface in `app.py` to ask questions like:
- "Show all suspicious activities from 192.168.0.23."
- "Summarize last night’s failed logins."

---

## Alert Types

| Alert Type                 | Trigger Condition                                |
| -------------------------- | ------------------------------------------------ |
| **Brute-force Attack**     | ≥5 failed logins from same IP/user within 2 mins |
| **Privilege Escalation**   | Sudden root/admin command after normal usage     |
| **Suspicious Access Time** | Login outside defined working hours              |
| **Frequent Reboots**       | >3 restarts within 10 mins                       |

---

## Extending the MVP
- Add real-time streaming (Kafka, ELK)
- Dashboard (Streamlit, Flask)
- SIEM integration
- Automated incident response

---