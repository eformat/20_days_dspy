import os

import dspy
from dspy.teleprompt import MIPROv2
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

print("--- Day 18: Instruction Optimization ---")


# 1. Define a simple signature for a task (e.g., paraphrasing)
class Paraphrase(dspy.Signature):
    """Paraphrase the given sentence to be more concise."""

    sentence: str = dspy.InputField()
    paraphrased_sentence: str = dspy.OutputField()


# 2. Create a dummy dataset
trainset_paraphrase = [
    dspy.Example(
        sentence="The extremely large and powerful automobile moved very quickly.",
        paraphrased_sentence="The large car sped away.",
    ).with_inputs("sentence"),
    dspy.Example(
        sentence="It is absolutely essential that you arrive on time.",
        paraphrased_sentence="Arrive on time.",
    ).with_inputs("sentence"),
    dspy.Example(
        sentence="Despite the fact that it was raining, we decided to go for a walk.",
        paraphrased_sentence="Despite the rain, we walked.",
    ).with_inputs("sentence"),
]

devset_paraphrase = [
    dspy.Example(
        sentence="In the event that you are unable to attend, please let us know.",
        paraphrased_sentence="If you can't attend, let us know.",
    ).with_inputs("sentence"),
    dspy.Example(
        sentence="He possesses a profound understanding of the subject matter.",
        paraphrased_sentence="He understands the subject well.",
    ).with_inputs("sentence"),
]

print("\n--- Step 1: Define a Metric for Paraphrasing ---")


# 3. Define a metric that assesses conciseness and semantic similarity (simplified)
#    In a real scenario, you'd use a more robust metric (e.g., ROUGE, BLEU, or an LLM-as-judge).
class AssessParaphrase(dspy.Signature):
    """Assess if the paraphrased text is concise and semantically similar to the original."""

    original_text: str = dspy.InputField()
    paraphrased_text: str = dspy.InputField()
    assessment_question: str = dspy.InputField(desc="A specific question to evaluate")
    assessment_answer: bool = dspy.OutputField(
        desc="True if the assessment question is true, False otherwise"
    )


def paraphrase_metric(example, pred, trace=None) -> float:
    lm_judge = dspy.Predict(AssessParaphrase)

    # Check conciseness (simplified: just word count comparison)
    original_words = len(example.sentence.split())
    paraphrased_words = len(pred.paraphrased_sentence.split())
    conciseness_score = (
        1.0 if paraphrased_words < original_words else 0.5
    )  # Encourage reduction

    # Check semantic similarity using an LLM-as-judge
    sem_sim_q = "Is the paraphrased text semantically equivalent to the original text?"
    sem_sim_assessment = lm_judge(
        original_text=example.sentence,
        paraphrased_text=pred.paraphrased_sentence,
        assessment_question=sem_sim_q,
    ).assessment_answer

    overall_score = (conciseness_score + (1.0 if sem_sim_assessment else 0.0)) / 2.0

    # For bootstrapping, return a strict boolean (e.g., if total score is above a threshold)
    if trace is not None:
        return overall_score >= 0.75  # Strict for bootstrapping
    return overall_score


print(f"Metric function defined: {paraphrase_metric.__name__}")

print("\n--- Step 2: Evaluate Baseline Paraphrase Program ---")
paraphrase_program = dspy.Predict(Paraphrase)
evaluator = dspy.Evaluate(
    devset=devset_paraphrase,
    metric=paraphrase_metric,
    num_threads=1,
    display_progress=True,
    display_table=True,
)
print("Evaluating baseline paraphrase program...")
baseline_score = evaluator(paraphrase_program)
print(f"Baseline Paraphrase Program Score: {baseline_score}")


print("\n--- Step 3: Optimize with MIPROv2 (Instruction Optimization) ---")

# 4. Initialize MIPROv2 optimizer


# Using 'light' auto-configuration for a quick run
# For 0-shot instruction optimization, set max_bootstrapped_demos=0, max_labeled_demos=0
teleprompter_mipro = MIPROv2(
    metric=paraphrase_metric,
    auto="light",  # Can choose between light, medium, and heavy
    num_threads=1,  # Keep low for local testing
    max_bootstrapped_demos=0,  # Focus on instruction optimization, not few-shot examples
    max_labeled_demos=0,
)

print("Compiling paraphrase program with MIPROv2 (Instruction Optimization)...")
optimized_paraphrase_program = teleprompter_mipro.compile(
    student=paraphrase_program.deepcopy(),
    trainset=trainset_paraphrase,
    requires_permission_to_run=False,  # Set to True if your metric involves LLM calls
)

print("\n--- Step 4: Evaluate Optimized Paraphrase Program ---")
print("Evaluating optimized paraphrase program...")
optimized_paraphrase_score = evaluator(optimized_paraphrase_program)
print(f"Optimized Paraphrase Program Score: {optimized_paraphrase_score}")

# Optional: Save the optimized program
optimized_paraphrase_program.save("optimized_paraphrase_mipro.json")
print(
    "\nOptimized Paraphrase program (MIPROv2) saved to optimized_paraphrase_mipro.json"
)

# Test the optimized program to see its new instruction
sample_sentence = "The individual who is responsible for the overall management of the project is currently out of the office."
optimized_prediction = optimized_paraphrase_program(sentence=sample_sentence)
print(f"\nOriginal: {sample_sentence}")
print(f"Optimized Paraphrase: {optimized_prediction.paraphrased_sentence}")

# Inspect history to see the new instruction (first interaction)
print(
    "\n--- Inspecting Optimized Program's History (last call with new instruction) ---"
)
dspy.inspect_history(n=1)
