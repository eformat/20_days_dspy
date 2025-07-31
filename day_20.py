import os

import dspy
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure language model
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.settings.configure(lm=lm, async_max_workers=4)  # Configure async capacity

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
