import os

import dspy

# Ensure LM is configured
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)

print("--- Day 11: Simple Tool Agent ---")


# 1. Define a simple tool function (e.g., a calculator)
def simple_calculator(expression: str) -> float:
    """Evaluates a mathematical expression."""
    try:
        # Using eval for simplicity, but be cautious in production due to security risks.
        # For production, consider using a safer math expression parser.
        return eval(expression)
    except Exception as e:
        return f"Error: {e}"


# 2. Wrap the function with dspy.Tool
#    Give it a descriptive name and description for the LM.
calculator_tool = dspy.Tool(
    name="Calculator",
    func=simple_calculator,
    desc="A tool to evaluate mathematical expressions and perform calculations.",
)


# 3. Define a signature for the agent's overall task
class SolveMathProblem(dspy.Signature):
    """Solve the given mathematical problem."""

    question: str = dspy.InputField()
    answer: float = dspy.OutputField(desc="numerical answer to the problem")


print("\n--- Building a dspy.ReAct Agent with a Calculator Tool ---")

# 4. Declare a dspy.ReAct module, passing the tool
#    ReAct will decide whether to use the tool and how.
math_agent = dspy.ReAct(SolveMathProblem, tools=[calculator_tool])

# 5. Run the agent with a question requiring calculation
question_1 = "What is 15.5 multiplied by 8 and then divided by 2?"
prediction_1 = math_agent(question=question_1)

print(f"Question: {question_1}")
print(f"Agent's Answer: {prediction_1.answer}")

question_2 = "Calculate (345 + 123) * 5 / 2."
prediction_2 = math_agent(question=question_2)

print(f"\nQuestion: {question_2}")
print(f"Agent's Answer: {prediction_2.answer}")

# Optional: Inspect history to see the ReAct trajectory (Thought, Action, Observation)
print("\n--- Inspecting LM History (last 2 ReAct trajectories) ---")
dspy.inspect_history(n=2)
