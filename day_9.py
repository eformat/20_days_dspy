import os
from typing import Dict, List

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

# Ensure LM is configured
# lm = dspy.LM(
#     "openai/gpt-4o-mini",
#     api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
# )
# dspy.configure(lm=lm)

print("--- Day 9: Document Summarization ---")


# 1. Define a signature for outlining a topic
class OutlineTopic(dspy.Signature):
    """Outline a thorough overview of a given topic, including title and main sections with subheadings."""

    topic = dspy.InputField()
    title = dspy.OutputField(desc="The main title for the overview")
    sections: List[str] = dspy.OutputField(desc="A list of main section headings")
    section_subheadings: Dict[str, List[str]] = dspy.OutputField(
        desc="A mapping from each section heading to its subheadings"
    )


# 2. Define a signature for drafting a single section
class DraftSection(dspy.Signature):
    """Draft a top-level section of an article given the topic, section heading, and subheadings."""

    topic = dspy.InputField(desc="The overall topic/title of the article")
    section_heading = dspy.InputField(desc="The main heading of the section to draft")
    section_subheadings: List[str] = dspy.InputField(
        desc="A list of subheadings for this section"
    )
    content: str = dspy.OutputField(desc="Markdown-formatted content for the section")


print("\n--- Building a Document Summarization Program ---")


# 3. Define the main ArticleDrafting module
class ArticleSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.build_outline = dspy.ChainOfThought(OutlineTopic)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        # Step 1: Build the outline
        outline = self.build_outline(topic=topic)

        all_sections_content = []
        # Step 2: Draft each section based on the outline
        for heading, subheadings in outline.section_subheadings.items():
            # Format headings for clarity in drafting prompt
            formatted_subheadings = (
                [f"### {sub}" for sub in subheadings] if subheadings else []
            )

            section_content = self.draft_section(
                topic=outline.title,
                section_heading=f"## {heading}",  # Pass main heading
                section_subheadings=formatted_subheadings,  # Pass subheadings
            ).content
            all_sections_content.append(section_content)

        # Combine all sections into a single summary/article
        full_article_content = "\n\n".join(all_sections_content)

        # The main output is the full article content, but we can also return title and sections for structure.
        return dspy.Prediction(
            title=outline.title,
            sections=all_sections_content,
            full_content=full_article_content,
        )


# 4. Instantiate and run the summarizer
summarizer_program = ArticleSummarizer()

#topic_to_summarize = "The History of Artificial Intelligence"
topic_to_summarize = "The History of Australia"
summary_prediction = summarizer_program(topic=topic_to_summarize)

print(f"Topic: {topic_to_summarize}")
print(f"Generated Title: {summary_prediction.title}")
print("\n--- Full Summarized Content ---")
print(summary_prediction.full_content)

# Optional: Inspect history
print("\n--- Inspecting LM History (last 2 full interactions) ---")
dspy.inspect_history(n=2)
