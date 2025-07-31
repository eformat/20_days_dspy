import os

import dspy

# Ensure LM is configured
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)

print("--- Day 13: Multi-Tool Agent ---")


# 1. Re-use or define multiple tool functions
# Calculator tool from Day 11
def simple_calculator(expression: str) -> float:
    """Evaluates a mathematical expression."""
    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {e}"


calculator_tool = dspy.Tool(
    name="Calculator",
    func=simple_calculator,
    desc="A tool to evaluate mathematical expressions and perform calculations.",
)

# Wikipedia Search tool from Day 12
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


# 2. Define a general problem-solving signature
class GeneralProblemSolver(dspy.Signature):
    """Answer the given question by leveraging available tools."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


print("\n--- Building a dspy.ReAct Agent with Multiple Tools ---")

# 3. Declare a dspy.ReAct module with both tools
multi_tool_agent = dspy.ReAct(
    GeneralProblemSolver, tools=[calculator_tool, wikipedia_search_tool]
)

# 4. Test with questions requiring different tools
question_1 = (
    "What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?"
)
prediction_1 = multi_tool_agent(question=question_1)

print(f"Question: {question_1}")
print(f"Agent's Answer: {prediction_1.answer}")

question_2 = "What is the capital of Peru and what is 500 divided by 25?"
prediction_2 = multi_tool_agent(question=question_2)

print(f"\nQuestion: {question_2}")
print(f"Agent's Answer: {prediction_2.answer}")

# Optional: Inspect history to see complex tool usage
print("\n--- Inspecting LM History (last 2 complex ReAct trajectories) ---")
dspy.inspect_history(n=2)
