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

#lm = dspy.LM(
#    "openai/gpt-4o-mini",
#    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
#)
#dspy.configure(lm=lm)

print("--- Day 1: Setting up DSPy and Simple Q&A ---")

# 2. Direct LM Call (Optional, for demonstration)
print("\n--- Direct LM Call ---")
response_direct = lm("Say hello!")
print(f"Direct LM Response: {response_direct[0]}")

# 3. Define a DSPy Signature
#    This specifies the input (question) and output (answer) fields.
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


print("\n--- Using dspy.Predict for Basic QA ---")

# 4. Declare a dspy.Predict module with the defined signature
#    dspy.Predict is the most basic module that predicts outputs based on a signature.
predict_qa = dspy.Predict(BasicQA)

# 5. Call the module with an input
question_1 = "What is the capital of France?"
prediction_1 = predict_qa(question=question_1)

# 6. Access the output
print(f"Question: {question_1}")
print(f"Predicted Answer: {prediction_1.answer}")

question_2 = "Who painted the Mona Lisa?"
prediction_2 = predict_qa(question=question_2)
print(f"\nQuestion: {question_2}")
print(f"Predicted Answer: {prediction_2.answer}")

# Optional: Inspect LM history
print("\n--- Inspecting LM History (last 2 calls) ---")
dspy.inspect_history(n=2)

# put another way
predict_qa_2 = dspy.Predict("question -> answer")

# 5. Call the module with an input
question_1 = "What is the capital of Egypt?"
prediction_1 = predict_qa_2(question=question_1)

# 6. Access the output
print(f"Question: {question_1}")
print(f"Predicted Answer: {prediction_1.answer}")

question_2 = "Who performed the song 'Bohemian Rhapsody'?"
prediction_2 = predict_qa_2(question=question_2)
print(f"\nQuestion: {question_2}")
print(f"Predicted Answer: {prediction_2.answer}")

# Optional: Inspect LM history
print("\n--- Inspecting LM History (last 2 calls) ---")
dspy.inspect_history(n=2)
