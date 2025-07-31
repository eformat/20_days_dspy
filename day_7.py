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

# # Ensure LM and RM are configured
# lm = dspy.LM(
#     "openai/gpt-4o-mini",
#     api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
# )
# dspy.configure(lm=lm)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

print("--- Day 7: Multi-Document RAG ---")


# 1. Re-use the RAGQA signature from Day 6
class RAGQA(dspy.Signature):
    """Answer questions based on the provided context."""

    context = dspy.InputField(desc="relevant passages from a knowledge base")
    question = dspy.InputField()
    answer = dspy.OutputField()


print("\n--- Building a Multi-Document RAG Program ---")


# 2. Modify the BasicRAG module to retrieve more passages
class MultiDocumentRAG(dspy.Module):
    def __init__(self, num_passages=5):  # Increase k to retrieve more documents
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(RAGQA)

    def forward(self, question):
        context = self.retrieve(question).passages
        # print(f"Retrieved {len(context)} passages.") # For debugging
        prediction = self.generate_answer(context=context, question=question)
        return prediction


# 3. Instantiate and run the Multi-Document RAG program
multi_doc_rag_program = MultiDocumentRAG(num_passages=5)  # Request 5 passages

question_multidoc_1 = (
    "Describe the life and major scientific contributions of Marie Curie."
)
prediction_multidoc_1 = multi_doc_rag_program(question=question_multidoc_1)

print(f"Question: {question_multidoc_1}")
print(
    f"Number of Retrieved Passages: {len(multi_doc_rag_program.retrieve(question_multidoc_1).passages)}"
)
print(f"Reasoning: {prediction_multidoc_1.reasoning}")
print(f"Answer: {prediction_multidoc_1.answer}")

question_multidoc_2 = "What are the main causes and effects of climate change?"
prediction_multidoc_2 = multi_doc_rag_program(question=question_multidoc_2)

print(f"\nQuestion: {question_multidoc_2}")
print(
    f"Number of Retrieved Passages: {len(multi_doc_rag_program.retrieve(question_multidoc_2).passages)}"
)
print(f"Reasoning: {prediction_multidoc_2.reasoning}")
print(f"Answer: {prediction_multidoc_2.answer}")

# Optional: Inspect history
print("\n--- Inspecting LM History (last 2 full interactions) ---")
dspy.inspect_history(n=2)
