import os

import dspy
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

# Ensure LM and RM are configured
# lm = dspy.LM(
#     "openai/gpt-4o-mini",
#     api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
# )
# dspy.configure(lm=lm)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

print("--- Day 8: Multi-Hop RAG ---")


# 1. Define signatures for query generation and answer generation
class GenerateSearchQuery(dspy.Signature):
    """Generate a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="previous context or notes")
    question = dspy.InputField()
    query = dspy.OutputField(desc="search query for retrieval")


class GenerateAnswer(dspy.Signature):
    """Answer the question based on the provided context."""

    context = dspy.InputField(desc="final gathered context")
    question = dspy.InputField()
    answer = dspy.OutputField()


print("\n--- Building a Multi-Hop RAG Program (Simplified Baleen) ---")


# Utility to deduplicate passages
def deduplicate(passages):
    seen = set()
    unique_passages = []
    for p in passages:
        if p not in seen:
            unique_passages.append(p)
            seen.add(p)
    return unique_passages


# 2. Define the Multi-Hop RAG module (similar to DSPy's SimplifiedBaleen example)
class MultiHopRAG(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()
        # Use a list of ChainOfThought for multiple query generations
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        # Store all queries to pass as context for subsequent query generations
        prev_queries_text = ""

        for hop in range(self.max_hops):
            query_gen_input_context = (
                f"Previous queries: {prev_queries_text}\n" if prev_queries_text else ""
            )
            query_gen_input_context += (
                f"Current context: {context}\n" if context else ""
            )

            # Generate a query based on accumulated context and question
            query = self.generate_query[hop](
                context=query_gen_input_context, question=question
            ).query
            prev_queries_text += f"Query {hop + 1}: {query}\n"  # Accumulate queries

            # Retrieve passages based on the generated query
            passages = self.retrieve(query).passages
            context = deduplicate(
                context + passages
            )  # Add new passages and deduplicate

        # Generate the final answer using all gathered context
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(
            context=context, answer=prediction.answer
        )  # Also return context for inspection


# 3. Instantiate and run the Multi-Hop RAG program
multihop_rag_program = MultiHopRAG(max_hops=2, passages_per_hop=3)

question_multihop_1 = "Which award did Gary Zukav's first book receive?"
prediction_multihop_1 = multihop_rag_program(question=question_multihop_1)

print(f"Question: {question_multihop_1}")
print(f"Final Context (first 3 passages):")
for i, passage in enumerate(prediction_multihop_1.context[:3]):
    print(f"  [{i + 1}] {passage}")
print(f"Answer: {prediction_multihop_1.answer}")


question_multihop_2 = "Who acted in the short film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of 'Hamlet'?"
prediction_multihop_2 = multihop_rag_program(question=question_multihop_2)

print(f"\nQuestion: {question_multihop_2}")
print(f"Final Context (first 3 passages):")
for i, passage in enumerate(prediction_multihop_2.context[:3]):
    print(f"  [{i + 1}] {passage}")
print(f"Answer: {prediction_multihop_2.answer}")

# Optional: Inspect history for multi-step reasoning
print("\n--- Inspecting LM History (last 2 full multi-hop interactions) ---")
dspy.inspect_history(n=2)
