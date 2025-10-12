"""
nlp_interface.py
- Handles natural language queries using LLMs (Hugging Face, LangChain).
"""
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


from transformers import pipeline
import pandas as pd

class NLPLLMInterface:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.qa_pipeline = pipeline('text2text-generation', model=model_name)
        self.llm = HuggingFacePipeline(pipeline=self.qa_pipeline)
        self.prompt = PromptTemplate(
            input_variables=["query", "logs", "alerts"],
            template="""
You are a cybersecurity log analysis assistant. Given the following logs and alerts, answer the user's query concisely and accurately.

Logs:
{logs}

Alerts:
{alerts}

User Query: {query}
Answer:
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def query(self, query: str, logs: pd.DataFrame, alerts: list) -> str:
        logs_str = logs[['timestamp','event','user','ip','message']].head(20).to_string(index=False)
        alerts_str = '\n'.join([f"[{a['alert_type']}] {a['message']}" for a in alerts])
        result = self.chain.run(query=query, logs=logs_str, alerts=alerts_str)
        return result
