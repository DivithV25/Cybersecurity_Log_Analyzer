"""
model_inference.py
- Loads transformer model, generates embeddings, classifies logs, outputs confidence scores.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd


from sklearn.neural_network import MLPClassifier
import numpy as np

class LogClassifier:
    def __init__(self, model_type='transformer', model_name='distilbert-base-uncased-finetuned-sst-2-english', dnn_model_path=None):
        self.model_type = model_type
        if model_type == 'transformer':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
        elif model_type == 'dnn':
            # Load a pre-trained DNN (scikit-learn MLPClassifier) from file if provided
            import joblib
            self.dnn = joblib.load(dnn_model_path) if dnn_model_path else None
        else:
            raise ValueError('Unsupported model_type')

    def classify_transformer(self, messages: pd.Series, batch_size: int = 8) -> pd.DataFrame:
        results = []
        for i in range(0, len(messages), batch_size):
            batch = messages.iloc[i:i+batch_size].tolist()
            inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                for j, s in enumerate(scores):
                    label = 'suspicious' if s[1] > 0.5 else 'normal'
                    confidence = float(s[1]) if label == 'suspicious' else float(s[0])
                    results.append({'classification': label, 'confidence': confidence})
        return pd.DataFrame(results)

    def classify_dnn(self, embeddings: np.ndarray) -> pd.DataFrame:
        # embeddings: shape (n_samples, n_features)
        preds = self.dnn.predict_proba(embeddings)
        results = []
        for s in preds:
            label = 'suspicious' if s[1] > 0.5 else 'normal'
            confidence = float(s[1]) if label == 'suspicious' else float(s[0])
            results.append({'classification': label, 'confidence': confidence})
        return pd.DataFrame(results)

    def classify(self, messages: pd.Series = None, embeddings: np.ndarray = None) -> pd.DataFrame:
        if self.model_type == 'transformer':
            if messages is None:
                raise ValueError('messages must be provided for transformer model')
            return self.classify_transformer(messages)
        elif self.model_type == 'dnn':
            if embeddings is None:
                raise ValueError('embeddings must be provided for dnn model')
            return self.classify_dnn(embeddings)
        else:
            raise ValueError('Unsupported model_type')
