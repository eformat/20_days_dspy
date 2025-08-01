import os
from typing import List

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
#lm = dspy.LM(
#    "openai/gpt-4o-mini",
#    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
#)
#dspy.configure(lm=lm)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

print("--- Day 10: Question-Based Summarization ---")

# 1. Define signatures for query generation, outlining, and section drafting
class GenerateSearchQuery(dspy.Signature):
    """Generate a simple search query that will help summarize a topic based on a question."""

    question = dspy.InputField()
    query = dspy.OutputField(desc="search query for retrieval")

class OutlineBasedOnQuestion(dspy.Signature):
    """Outline a summary for a given topic and question, using provided context."""

    context = dspy.InputField(desc="retrieved relevant passages")
    question = dspy.InputField()
    title = dspy.OutputField(desc="A concise title for the summary")
    sections: List[str] = dspy.OutputField(
        desc="Main section headings relevant to the question"
    )

class DraftSummarySection(dspy.Signature):
    """Draft a section of a summary, focusing on relevance to the question and given context."""

    context = dspy.InputField(desc="Relevant passages for this section")
    question = dspy.InputField()
    section_heading = dspy.InputField()
    content: str = dspy.OutputField(
        desc="Markdown-formatted content for the summary section"
    )

print("\n--- Building a Question-Based Summarizer Program ---")

# 2. Define the QuestionBasedSummarizer module
class QuestionBasedSummarizer(dspy.Module):
    def __init__(self, num_search_results=5):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.retrieve = dspy.Retrieve(k=num_search_results)
        self.outline_summary = dspy.ChainOfThought(OutlineBasedOnQuestion)
        self.draft_section = dspy.ChainOfThought(DraftSummarySection)

    def forward(self, topic, question):
        # Step 1: Generate a search query based on the topic and question
        query = self.generate_query(question=question).query

        # Step 2: Retrieve relevant passages
        context = self.retrieve(query).passages

        # Step 3: Outline the summary based on retrieved context and question
        outline = self.outline_summary(context=context, question=question)

        summary_sections = []
        # Step 4: Draft each summary section
        for section_heading in outline.sections:
            # For simplicity, we'll pass the whole context to each section draft.
            # More advanced would involve selecting relevant context for each section.
            section_content = self.draft_section(
                context=context, question=question, section_heading=section_heading
            ).content
            summary_sections.append(section_content)

        full_summary_content = f"# {outline.title}\n\n" + "\n\n".join(summary_sections)

        return dspy.Prediction(
            title=outline.title,
            sections=summary_sections,
            full_summary=full_summary_content,
        )

# 3. Instantiate and run the question-based summarizer
q_based_summarizer = QuestionBasedSummarizer(num_search_results=7)

topic_qs = "Artificial Intelligence"
question_qs = "What are the recent advancements in AI and its ethical implications?"
summary_prediction_qs = q_based_summarizer(topic=topic_qs, question=question_qs)

print(f"Topic: {topic_qs}")
print(f"Question: {question_qs}")
print("\n--- Generated Question-Based Summary ---")
print(summary_prediction_qs.full_summary)

# Optional: Inspect history
print("\n--- Inspecting LM History (last 2 full interactions) ---")
dspy.inspect_history(n=2)
