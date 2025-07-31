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


# Ensure LM is configured
#lm = dspy.LM(
#    "openai/gpt-4o-mini",
#    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
#)
#dspy.configure(lm=lm)

print("--- Day 3: Math Problem Solver ---")


# 1. Define a signature for math problems with a float output type
class MathQA(dspy.Signature):
    """Solve the given math question and output the numerical answer."""

    question = dspy.InputField()
    answer: float = dspy.OutputField(desc="numerical answer")


print("\n--- Using dspy.ChainOfThought for Math Problems ---")

# 2. Declare a ChainOfThought module for math
math_solver = dspy.ChainOfThought(MathQA)

# 3. Test with a simple math question
question_math_1 = (
    "If a train travels at 60 miles per hour for 2.5 hours, how far does it travel?"
)
prediction_math_1 = math_solver(question=question_math_1)

print(f"Question: {question_math_1}")
print(f"Reasoning: {prediction_math_1.reasoning}")
print(f"Answer: {prediction_math_1.answer} miles")

# 4. Test with a slightly more complex math question
question_math_2 = "Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?"
prediction_math_2 = math_solver(question=question_math_2)

print(f"\nQuestion: {question_math_2}")
print(f"Reasoning: {prediction_math_2.reasoning}")
print(f"Answer: {prediction_math_2.answer} apples")

# 5. Example from DSPy docs for two dice toss
question_math_3 = (
    "Two dice are tossed. What is the probability that the sum equals two?"
)
prediction_math_3 = math_solver(question=question_math_3)

print(f"\nQuestion: {question_math_3}")
print(f"Reasoning: {prediction_math_3.reasoning}")
print(f"Answer: {prediction_math_3.answer}")

# Optional: Inspect history
print("\n--- Inspecting LM History (last 3 calls) ---")
dspy.inspect_history(n=3)
