import os

import dspy
import openai
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


# Ensure LM is configured (from Day 1 setup)
#lm = dspy.LM(
#    "openai/gpt-4o-mini",
#    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
#)
#dspy.configure(lm=lm)

print("--- Day 2: Chain of Thought ---")


# 1. Define the same basic QA signature
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


print("\n--- Using dspy.ChainOfThought for Enhanced QA ---")

# 2. Declare a dspy.ChainOfThought module with the signature
#    ChainOfThought automatically adds a 'reasoning' field to the prompt.
cot_qa = dspy.ChainOfThought(BasicQA)

# 3. Call the module with a question
question_cot = "What is the primary function of chlorophyll in plants?"
prediction_cot = cot_qa(question=question_cot)

# 4. Access both reasoning and the final answer
print(f"Question: {question_cot}")
print(f"Reasoning: {prediction_cot.reasoning}")
print(f"Answer: {prediction_cot.answer}")

question_cot_2 = "Name the largest organ in the human body."
prediction_cot_2 = cot_qa(question=question_cot_2)
print(f"\nQuestion: {question_cot_2}")
print(f"Reasoning: {prediction_cot_2.reasoning}")
print(f"Answer: {prediction_cot_2.answer}")

# Optional: Inspect LM history to see the CoT prompt
print("\n--- Inspecting LM History (last 2 calls to see CoT format) ---")
dspy.inspect_history(n=2)
