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


# Ensure LM is configured
#lm = dspy.LM(
#    "openai/gpt-4o-mini",
#    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
#)
#dspy.configure(lm=lm)

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
