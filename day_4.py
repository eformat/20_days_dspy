import os
from typing import Literal

import dspy

# Ensure LM is configured
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)

print("--- Day 4: Sentiment Classifier ---")


# 1. Define a signature for sentiment classification with Literal and multiple outputs
class ClassifySentiment(dspy.Signature):
    """Classify the sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField(
        desc="a score between 0.0 and 1.0 indicating certainty"
    )


print("\n--- Using dspy.Predict for Sentiment Classification ---")

# 2. Declare a dspy.Predict module for classification
sentiment_classifier = dspy.Predict(
    "sentence -> sentiment: Literal['positive', 'negative', 'neutral'], confidence: float"
)
# OR
# sentiment_classifier = dspy.Predict(ClassifySentiment)


# 3. Test with various sentences
sentence_1 = "This movie was absolutely brilliant and captivating from start to finish!"
prediction_1 = sentiment_classifier(sentence=sentence_1)
print(f"Sentence: '{sentence_1}'")
print(f"Sentiment: {prediction_1.sentiment}, Confidence: {prediction_1.confidence}")

sentence_2 = "The food was okay, but the service was incredibly slow."
prediction_2 = sentiment_classifier(sentence=sentence_2)
print(f"\nSentence: '{sentence_2}'")
print(f"Sentiment: {prediction_2.sentiment}, Confidence: {prediction_2.confidence}")

sentence_3 = "I have no strong feelings about this."
prediction_3 = sentiment_classifier(sentence=sentence_3)
print(f"\nSentence: '{sentence_3}'")
print(f"Sentiment: {prediction_3.sentiment}, Confidence: {prediction_3.confidence}")

sentence_4 = "This book was super fun to read, though not the last chapter."
prediction_4 = sentiment_classifier(sentence=sentence_4)
print(f"\nSentence: '{sentence_4}'")
print(f"Sentiment: {prediction_4.sentiment}, Confidence: {prediction_4.confidence}")

# Optional: Inspect history
print("\n--- Inspecting LM History (last 4 calls) ---")
dspy.inspect_history(n=4)
