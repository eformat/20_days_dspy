import os
from typing import Dict, List

import dspy
from pydantic import BaseModel

import mlflow

mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("20_days_dspy")
mlflow.dspy.autolog(
    log_compiles=True,    # Track optimization process
    log_evals=True,       # Track evaluation results
    log_traces_from_compile=True  # Track program traces during optimization
)

# 1. Configure the Language Model
#    Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key.
#    You can also set it as an environment variable: export OPENAI_API_KEY='your_key_here'
#    If using a local LM with Ollama or SGLang, adjust api_base and api_key accordingly.

LLM_URL=os.getenv('LLM_URL', 'http://localhost:8080/v1')
API_KEY=os.getenv('API_KEY', 'fake')
LLM_MODEL=os.getenv('LLM_MODEL', 'openai/models/Llama-3.2-3B-Instruct-Q8_0.gguf')
MAX_TOKENS=os.getenv('MAX_TOKENS', 3000)
TEMPERATURE=os.getenv('TEMPERATURE', 0.2)
dspy.enable_logging()

lm = dspy.LM(model=LLM_MODEL,
             api_base=LLM_URL,
             api_key=API_KEY,
             temperature=TEMPERATURE,
             model_type='chat',
             stream=False)
dspy.configure(lm=lm)
dspy.settings.configure(track_usage=True)

# Ensure LM is configured
#lm = dspy.LM(
#    "openai/gpt-4o-mini",
#    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
#)
#dspy.configure(lm=lm)

print("--- Day 5: Information Extraction to Pydantic---")

class Entity(BaseModel):
    name: str
    entity_type: str

class ArticleInfo(BaseModel):
    title: str
    headings: List[str]
    entities: List[Entity]

# 1. Define a signature for extracting structured information
class ExtractArticleInfo(dspy.Signature):
    """Extract structured information from a given text, including title, headings, and entities."""

    text: str = dspy.InputField()
    article_info: ArticleInfo = dspy.OutputField(desc="The main title of the content")

print("\n--- Using dspy.Predict for Information Extraction ---")

# 2. Declare a dspy.Predict module for information extraction
info_extractor = dspy.Predict(ExtractArticleInfo)

# 3. Test with a sample text
sample_text_1 = """
Apple Inc. announced its latest iPhone 16 today at the annual WWDC conference.
The CEO, Tim Cook, highlighted its revolutionary new features, including an
AI-powered camera and a titanium frame, in a press release. The event took place
in Cupertino, California.
"""
prediction_1 = info_extractor(text=sample_text_1)

print(f"Original Text:\n{sample_text_1}\n")
print(f"Extracted Title: {prediction_1.article_info.title}")
print(f"Extracted Headings: {prediction_1.article_info.headings}")
print(f"Extracted Entities: {prediction_1.article_info.entities}")

sample_text_2 = """
The highly anticipated sequel, 'Dune: Part Two', starring Timothée Chalamet and Zendaya,
was released on March 1, 2024. Directed by Denis Villeneuve, the film grossed over
$711 million worldwide.
"""
prediction_2 = info_extractor(text=sample_text_2)

print(f"\nOriginal Text:\n{sample_text_2}\n")
print(f"Extracted Title: {prediction_2.article_info.title}")
print(f"Extracted Headings: {prediction_2.article_info.headings}")
print(f"Extracted Entities: {prediction_2.article_info.entities}")

# Optional: Inspect history
print("\n--- Inspecting LM History (last 2 calls) ---")
dspy.inspect_history(n=2)
