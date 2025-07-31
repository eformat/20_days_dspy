import os

import dspy

# Ensure LM is configured
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)

print("--- Day 12: Search Agent ---")

# 1. Define a search tool (using ColBERTv2 for Wikipedia abstracts, or you can use Tavily for web search)
#    For simplicity, we'll re-use the ColBERTv2 from Week 2.
#    If you want a real web search, you'd integrate a library like TavilyClient.
#    E.g., from tavily import TavilyClient; search_client = TavilyClient(api_key="<YOUR_TAVILY_API_KEY>")
#    def web_search(query: str) -> list[str]: return [r["content"] for r in search_client.search(query)["results"]]
#    search_tool = dspy.Tool(web_search, name="WebSearch", description="A tool for searching the web.")

colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")


def search_wikipedia_tool(query: str) -> list[str]:
    """Retrieves abstracts from Wikipedia using ColBERTv2."""
    results = colbert(query, k=3)
    return [x["text"] for x in results]


wikipedia_search_tool = dspy.Tool(
    name="WikipediaSearch",
    func=search_wikipedia_tool,
    desc="A tool for searching Wikipedia for factual information.",
)


# 2. Define the signature for a factual question answering agent
class FactualQA(dspy.Signature):
    """Answer factual questions accurately."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


print("\n--- Building a dspy.ReAct Agent with a Search Tool ---")

# 3. Declare a dspy.ReAct module with the search tool
search_agent = dspy.ReAct(FactualQA, tools=[wikipedia_search_tool])

# 4. Run the agent with a question requiring external search
question_1 = "Which baseball team does Shohei Ohtani play for?"
prediction_1 = search_agent(question=question_1)

print(f"Question: {question_1}")
print(f"Agent's Answer: {prediction_1.answer}")

question_2 = "What year was the Golden Gate Bridge completed?"
prediction_2 = search_agent(question=question_2)

print(f"\nQuestion: {question_2}")
print(f"Agent's Answer: {prediction_2.answer}")

# Optional: Inspect history to see search queries and observations
print("\n--- Inspecting LM History (last 2 ReAct trajectories) ---")
dspy.inspect_history(n=2)
