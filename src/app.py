"""
app.py
- Entry point for the Cybersecurity Log Analyzer MVP.
- Usage: python app.py --input sample.log
"""
import argparse
import pandas as pd
from data_pipeline import LogDataPipeline
from model_inference import LogClassifier
from alert_engine import AlertEngine
from nlp_interface import NLPLLMInterface


def main():

    parser = argparse.ArgumentParser(description="Cybersecurity Log Analyzer MVP")
    parser.add_argument('--input', type=str, required=True, help='Path to log file')
    parser.add_argument('--query', type=str, help='Natural language query (optional)')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'dnn'], help='Model type')
    args = parser.parse_args()

    print("[1] Parsing and preprocessing logs...")
    pipeline = LogDataPipeline()
    df = pipeline.process_log_file(args.input)

    print("[2] Running AI model inference...")
    classifier = LogClassifier(model_type=args.model)
    results = classifier.classify(messages=df['message'])
    df = df.reset_index(drop=True).join(results)

    print("[3] Generating alerts...")
    alert_engine = AlertEngine()
    alerts = alert_engine.generate_alerts(df)
    print(f"[DONE] {len(alerts)} alerts generated.")

    print("\n=== Anomaly Detection Results ===")
    print(df[['timestamp','event','user','ip','classification','confidence']].head(20))
    print(f"\nTotal alerts generated: {len(alerts)}")
    for alert in alerts[:3]:
        print(f"\nALERT: {alert['alert_type']} | {alert['message']}")

    # NLP interface for natural language queries
    if args.query:
        nlp = NLPLLMInterface()
        response = nlp.query(args.query, df, alerts)
        print(f"\nLLM Response:\n{response}")

if __name__ == "__main__":
    main()
