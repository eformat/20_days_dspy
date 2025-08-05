import os

import dspy
from dspy.evaluate import Evaluate  # Import Evaluate for the judge
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

# Ensure LM is configured
# lm = dspy.LM(
#     "openai/gpt-4o-mini",
#     api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
# )
# dspy.configure(lm=lm)

print("--- Day 19: Advanced Refinement with Refine ---")


# 1. Define a signature for a simple answer generation
class AnswerQuestion(dspy.Signature):
    """Answer the question clearly and concisely."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# 2. Define a judgment signature for evaluation
class AssessAnswer(dspy.Signature):
    """Assess the quality of an answer along specific dimensions."""

    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    assessment_question: str = dspy.InputField(
        desc="A specific question to evaluate the answer"
    )
    assessment_result: bool = dspy.OutputField(
        desc="True if assessment is positive, False otherwise"
    )
    feedback: str = dspy.OutputField(desc="Specific feedback for improvement if False")


print("\n--- Step 1: Define a Refinement-Driven Metric (LLM-as-Judge) ---")


# 3. Define a reward function that uses an LLM-as-Judge and provides feedback
def conciseness_and_accuracy_reward(args, pred: dspy.Prediction) -> float:
    question = args["question"]  # Access original input args passed to the module
    answer = pred.answer

    judge = dspy.ChainOfThought(AssessAnswer)  # Use ChainOfThought for assessment

    # Assess conciseness
    concise_assessment = judge(
        question=question,
        answer=answer,
        assessment_question="Is the answer concise (under 20 words)?",
    ).assessment_result

    # Assess accuracy (simplified, assuming LM has factual knowledge or context is implicitly used)
    accurate_assessment = judge(
        question=question,
        answer=answer,
        assessment_question="Is the answer factually accurate and directly addresses the question?",
    ).assessment_result

    score = (1.0 if concise_assessment else 0.0) + (1.0 if accurate_assessment else 0.0)

    # Refine specifically uses this score for its internal optimization.
    # It passes 'trace' when it's doing internal bootstrapping/refinement, so we can be more strict.
    # When trace is None, it's a final evaluation.
    # For Refine, a simple score (0 or 1) per attempt works.
    return score / 2.0  # Return a score between 0 and 1


print(f"Reward function defined: {conciseness_and_accuracy_reward.__name__}")

print("\n--- Step 2: Implement Refine Module ---")

# 4. Declare a dspy.Refine module
#    module: The core module to be refined.
#    N: Maximum number of attempts.
#    reward_fn: The function that scores predictions and provides feedback.
#    threshold: If a prediction's score meets this, refinement stops early.
#    fail_count: Number of consecutive failures before giving up (optional).
refine_qa_program = dspy.Refine(
    module=dspy.ChainOfThought(AnswerQuestion),
    N=3,  # Try up to 3 times
    reward_fn=conciseness_and_accuracy_reward,
    threshold=0.8,  # Stop if score >= 0.8 (e.g., both concise and accurate)
    fail_count=1,  # Stop after 1 error (e.g., if LM gets stuck)
)

# 5. Run the refined program
question_refine_1 = "Briefly explain the concept of photosynthesis."
prediction_refine_1 = refine_qa_program(question=question_refine_1)

print(f"Question: {question_refine_1}")
print(f"Refined Answer: {prediction_refine_1.answer}")

question_refine_2 = "What is the capital of New Zealand? Be extremely concise."
prediction_refine_2 = refine_qa_program(question=question_refine_2)

print(f"\nQuestion: {question_refine_2}")
print(f"Refined Answer: {prediction_refine_2.answer}")

# Optional: Inspect history to see the refinement process
print("\n--- Inspecting LM History (last 2 refinement attempts) ---")
dspy.inspect_history(n=2)
