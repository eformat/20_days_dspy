import os

import dspy

# Ensure LM and RM are configured
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

print("--- Day 17: Few-Shot Optimization ---")


# 1. Define RAG signature (from Day 6)
class RAGQA(dspy.Signature):
    """Answer questions based on the provided context."""

    context: str = dspy.InputField(desc="relevant passages from a knowledge base")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# 2. Define a RAG program (from Day 6)
class BasicRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(RAGQA)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return prediction


# 3. Create dummy dataset for RAG (simulating HotPotQA examples)
# In a real scenario, use dspy.datasets.HotPotQA
trainset_rag = [
    dspy.Example(
        question="Who was the first person to walk on the moon?",
        answer="Neil Armstrong",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the capital of Australia?", answer="Canberra"
    ).with_inputs("question"),
    dspy.Example(question="When did World War II end?", answer="1945").with_inputs(
        "question"
    ),
    dspy.Example(
        question="What is the chemical formula for sulfuric acid?", answer="H2SO4"
    ).with_inputs("question"),
]

devset_rag = [
    dspy.Example(
        question="What is the highest mountain in North America?", answer="Denali"
    ).with_inputs("question"),
    dspy.Example(
        question="Who invented the light bulb?", answer="Thomas Edison"
    ).with_inputs("question"),
]

print("\n--- Step 1: Define a Metric for RAG ---")


# 4. Define a RAG-specific metric (answer exact match from documentation)
def answer_exact_match(example, pred, trace=None):
    """Evaluates if the predicted answer exactly matches the gold answer (case-insensitive)."""
    return example.answer.lower() == pred.answer.lower()


print(f"Metric function defined: {answer_exact_match.__name__}")

print("\n--- Step 2: Evaluate Baseline RAG Program ---")
rag_program = BasicRAG()
evaluator = dspy.Evaluate(
    devset=devset_rag,
    metric=answer_exact_match,
    num_threads=1,
    display_progress=True,
    display_table=True,
)
print("Evaluating baseline RAG program...")
baseline_score = evaluator(rag_program)
print(f"Baseline RAG Program Score: {baseline_score}")

print("\n--- Step 3: Optimize RAG with BootstrapFewShotWithRandomSearch ---")

# 5. Initialize BootstrapFewShotWithRandomSearch optimizer
#    num_candidate_programs: how many random bootstrapped programs to try.
#    num_threads: parallelism for evaluation during optimization.
from dspy.teleprompt import (
    BootstrapFewShotWithRandomSearch,  # Note: from dspy.teleprompt for optimizers
)

optimizer_config = dict(
    max_bootstrapped_demos=2,  # Self-generate up to 2 demos per predictor
    max_labeled_demos=0,  # Don't use labeled demos for bootstrapping (rely on teacher's trace)
    num_candidate_programs=3,  # Try 3 different random seeds for bootstrapping
    num_threads=1,  # Keep low for local testing
)

# Use a deepcopy of the RAG program to optimize.
# The `teacher` argument can be left as default (the program itself) or a more capable LM.
teleprompter_rag = BootstrapFewShotWithRandomSearch(
    metric=answer_exact_match, **optimizer_config
)

print("Compiling RAG program with BootstrapFewShotWithRandomSearch...")
optimized_rag_program = teleprompter_rag.compile(
    student=rag_program.deepcopy(),
    trainset=trainset_rag,
    valset=devset_rag,  # valset used by random search to pick best program
)

print("\n--- Step 4: Evaluate Optimized RAG Program ---")
print("Evaluating optimized RAG program...")
optimized_rag_score = evaluator(optimized_rag_program)
print(f"Optimized RAG Program Score: {optimized_rag_score}")

# Optional: Save and load the optimized program
optimized_rag_program.save("optimized_rag.json")
print("\nOptimized RAG program saved to optimized_rag.json")

loaded_rag_program = BasicRAG()  # Recreate the class structure
loaded_rag_program.load("optimized_rag.json")
print("Optimized RAG program loaded from optimized_rag.json")

# Verify that demos were loaded (they become part of the program's state)
# This check is illustrative, internal demos structure can vary per module/optimizer.
# The demos attribute may not exist in all DSPy versions, so we'll check for it safely
if hasattr(optimized_rag_program, "demos") and hasattr(loaded_rag_program, "demos"):
    print(f"Number of demos in optimized program: {len(optimized_rag_program.demos)}")
    print(f"Number of demos in loaded program: {len(loaded_rag_program.demos)}")
    assert len(optimized_rag_program.demos) == len(loaded_rag_program.demos)
else:
    print(
        "Demo attribute check skipped - demos structure may vary in this DSPy version"
    )
