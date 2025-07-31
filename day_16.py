import os

import dspy

# Ensure LM is configured
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)

print("--- Day 16: Metrics & Optimization Basics ---")


# 1. Define a simple signature for QA
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")


# 2. Create a dummy dataset (usually loaded from file)
# In a real scenario, you'd load from HotPotQA, a CSV, etc.
trainset = [
    dspy.Example(
        question="What is the capital of Canada?", answer="Ottawa"
    ).with_inputs("question"),
    dspy.Example(
        question="Who wrote 'To Kill a Mockingbird'?", answer="Harper Lee"
    ).with_inputs("question"),
    dspy.Example(
        question="What is the highest mountain in Africa?", answer="Mount Kilimanjaro"
    ).with_inputs("question"),
    dspy.Example(
        question="What is the chemical symbol for water?", answer="H2O"
    ).with_inputs("question"),
    dspy.Example(
        question="How many planets are in our solar system?", answer="8"
    ).with_inputs("question"),
]

devset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs(
        "question"
    ),
    dspy.Example(
        question="What is the largest ocean on Earth?", answer="Pacific Ocean"
    ).with_inputs("question"),
]

print("\n--- Step 1: Define a Metric ---")


# 3. Define a simple exact match metric
def exact_match_metric(
    example, pred, trace=None
) -> bool:  # trace is optional for some optimizers
    """Checks if the predicted answer exactly matches the gold answer (case-insensitive)."""
    return example.answer.lower() == pred.answer.lower()


print(f"Metric function defined: {exact_match_metric.__name__}")

print("\n--- Step 2: Evaluate a Baseline Program ---")

# 4. Define a baseline program (unoptimized)
baseline_program = dspy.Predict(BasicQA)

# 5. Set up the evaluator
evaluator = dspy.Evaluate(
    devset=devset,
    metric=exact_match_metric,
    num_threads=1,  # For local testing, keep low. Increase for faster eval.
    display_progress=True,
    display_table=True,  # Show results table
)

# 6. Run evaluation for the baseline program
print("Evaluating baseline program...")
baseline_score = evaluator(baseline_program)
print(f"Baseline Program Score: {baseline_score}")

print("\n--- Step 3: Basic Optimization with BootstrapFewShot ---")

# 7. Initialize BootstrapFewShot optimizer
#    max_bootstrapped_demos: how many examples to self-generate for each step.
#    max_labeled_demos: how many provided labeled examples to use.
optimizer = dspy.BootstrapFewShot(
    metric=exact_match_metric,
    max_bootstrapped_demos=2,  # Self-generate 2 good examples
    max_labeled_demos=2,  # Use 2 provided labeled examples
    max_rounds=1,  # Only one round of bootstrapping
    max_errors=5,  # Allow up to 5 errors during bootstrapping
)

# 8. Compile the program using the optimizer and training data
print("Compiling program with BootstrapFewShot...")
optimized_program = optimizer.compile(
    student=baseline_program.deepcopy(),  # It's good practice to pass a deepcopy
    trainset=trainset,
)

print("\n--- Step 4: Evaluate the Optimized Program ---")
print("Evaluating optimized program...")
optimized_score = evaluator(optimized_program)
print(f"Optimized Program Score: {optimized_score}")

# Optional: Inspect history of the optimized program
print("\n--- Inspecting Optimized Program's History (last call with demos) ---")
# Call it once to populate history if not done by evaluation
question_optimized_test = "What is the capital of Italy?"
optimized_program(question=question_optimized_test)
dspy.inspect_history(
    n=1
)  # Look at the last interaction, should include few-shot examples
