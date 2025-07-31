import os

import dspy

# Ensure LM is configured (from Week 1 setup)
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)

# 1. Configure a Retrieval Model (ColBERTv2)
#    This is a public ColBERTv2 endpoint for Wikipedia abstracts.
colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

print("--- Day 6: Basic RAG ---")


# 2. Define a RAG signature
#    The context field is crucial for RAG, it will be populated by the retriever.
class RAGQA(dspy.Signature):
    """Answer questions based on the provided context."""

    context = dspy.InputField(desc="relevant passages from a knowledge base")
    question = dspy.InputField()
    answer = dspy.OutputField()


print("\n--- Building a Basic RAG Program ---")


# 3. Define a DSPy Module for RAG
class BasicRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)  # Retrieve top-k passages
        self.generate_answer = dspy.ChainOfThought(
            RAGQA
        )  # Generate answer using ChainOfThought

    def forward(self, question):
        # 4. Perform retrieval
        context = self.retrieve(question).passages
        # 5. Generate answer using the retrieved context
        prediction = self.generate_answer(context=context, question=question)
        return prediction


# 6. Instantiate and run the Basic RAG program
rag_program = BasicRAG()

question_rag_1 = "When was the first FIFA World Cup held?"
prediction_rag_1 = rag_program(question=question_rag_1)

print(f"Question: {question_rag_1}")
print(f"Retrieved Context (first 2 passages):")
for i, passage in enumerate(rag_program.retrieve(question_rag_1).passages[:2]):
    print(f"  [{i + 1}] {passage}")
print(f"Reasoning: {prediction_rag_1.reasoning}")
print(f"Answer: {prediction_rag_1.answer}")

question_rag_2 = "What is the capital of Portugal?"
prediction_rag_2 = rag_program(question_rag_2)

print(f"\nQuestion: {question_rag_2}")
print(f"Retrieved Context (first 2 passages):")
for i, passage in enumerate(rag_program.retrieve(question_rag_2).passages[:2]):
    print(f"  [{i + 1}] {passage}")
print(f"Reasoning: {prediction_rag_2.reasoning}")
print(f"Answer: {prediction_rag_2.answer}")

# Optional: Inspect history
print("\n--- Inspecting LM History (last 2 full interactions including retrieval) ---")
dspy.inspect_history(n=2)
