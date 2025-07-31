import os
from typing import Dict, List

import dspy
from pydantic import BaseModel

# Ensure LM is configured
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)

print("--- Day 5: Information Extraction to Pydantic---")


class Entity(BaseModel):
    name: str
    entity_type: str


class ArticleInfo(BaseModel):
    title: str
    headings: List[str]
    entities: List[Entity]


# 1. Define a signature for extracting structured information
class ExtractArticleInfo(dspy.Signature):
    """Extract structured information from a given text, including title, headings, and entities."""

    text: str = dspy.InputField()
    article_info: ArticleInfo = dspy.OutputField(desc="The main title of the content")


print("\n--- Using dspy.Predict for Information Extraction ---")

# 2. Declare a dspy.Predict module for information extraction
info_extractor = dspy.Predict(ExtractArticleInfo)

# 3. Test with a sample text
sample_text_1 = """
Apple Inc. announced its latest iPhone 16 today at the annual WWDC conference.
The CEO, Tim Cook, highlighted its revolutionary new features, including an
AI-powered camera and a titanium frame, in a press release. The event took place
in Cupertino, California.
"""
prediction_1 = info_extractor(text=sample_text_1)

print(f"Original Text:\n{sample_text_1}\n")
print(f"Extracted Title: {prediction_1.article_info.title}")
print(f"Extracted Headings: {prediction_1.article_info.headings}")
print(f"Extracted Entities: {prediction_1.article_info.entities}")

sample_text_2 = """
The highly anticipated sequel, 'Dune: Part Two', starring Timothée Chalamet and Zendaya,
was released on March 1, 2024. Directed by Denis Villeneuve, the film grossed over
$711 million worldwide.
"""
prediction_2 = info_extractor(text=sample_text_2)

print(f"\nOriginal Text:\n{sample_text_2}\n")
print(f"Extracted Title: {prediction_2.article_info.title}")
print(f"Extracted Headings: {prediction_2.article_info.headings}")
print(f"Extracted Entities: {prediction_2.article_info.entities}")

# Optional: Inspect history
print("\n--- Inspecting LM History (last 2 calls) ---")
dspy.inspect_history(n=2)
