import os

import dspy
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from rich.pretty import pprint

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


# Configure language model
# lm = dspy.LM(
#     "openai/gpt-4o-mini",
#     api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
# )
# dspy.settings.configure(lm=lm, async_max_workers=4)  # Configure async capacity

print("--- Day 20: Deployment ---")


# Define DSPy program
class SimpleQA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


dspy_program = dspy.ChainOfThought(SimpleQA)

# Apply dspy.asyncify for async execution
dspy_program = dspy.asyncify(dspy_program)

app = FastAPI(
    title="DSPy Program API",
    description="A simple API serving a DSPy Chain of Thought program",
    version="1.0.0",
)


# Define request model for better documentation and validation
class Question(BaseModel):
    text: str


@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)  # Use await for async call
        return {"status": "success", "data": result.toDict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # Example curl command:
    # curl -X POST "http://127.0.0.1:8000/predict" \
    #      -H "Content-Type: application/json" \
    #      -d '{"text": "What is the capital of France?"}'
