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
# lm = dspy.LM(
#     "openai/gpt-4o-mini",
#     api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
# )
# dspy.configure(lm=lm)

print("--- Day 15: Specialized Domain Agent ---")

# --- Hypothetical Medical Domain Tools ---
# These are simplified for demonstration; in a real scenario, they would
# query databases, APIs, or specialized knowledge bases.

medical_definitions = {
    "hypertension": "High blood pressure, a condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
    "diabetes": "A chronic medical condition in which the body either doesn't produce enough insulin or can't effectively use the insulin it does produce, leading to high blood sugar levels.",
    "migraine": "A severe throbbing headache or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound.",
}


def get_medical_definition(term: str) -> str:
    """Provides a simplified definition for common medical terms."""
    definition = medical_definitions.get(
        term.lower(), "Definition not found for that term."
    )
    return f"Definition of {term}: {definition}"


medical_def_tool = dspy.Tool(
    name="MedicalDefinitionLookup",
    func=get_medical_definition,
    desc="A tool to look up simplified definitions of medical terms. Input should be a single medical term.",
)

# --- Hypothetical Financial Domain Tools ---
stock_prices = {"AAPL": 170.00, "GOOG": 150.00, "MSFT": 420.00}


def get_stock_price(ticker: str) -> float:
    """Fetches the current simulated stock price for a given ticker symbol."""
    price = stock_prices.get(ticker.upper())
    if price:
        return price
    return f"Stock price not found for ticker: {ticker.upper()}"


stock_price_tool = dspy.Tool(
    name="StockPriceChecker",
    func=get_stock_price,
    desc="A tool to check the current simulated stock price of a given company ticker symbol (e.g., AAPL, GOOG, MSFT).",
)


# 2. Define a general signature for a domain-specific agent
class DomainSpecificQA(dspy.Signature):
    """Answer questions within a specialized domain by using relevant tools."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


print("\n--- Building a Specialized Domain Agent with Multiple Domain Tools ---")

# 3. Declare a dspy.ReAct module with all domain-specific tools
#    The agent will decide which domain's tool to use based on the question.
domain_agent = dspy.ReAct(DomainSpecificQA, tools=[medical_def_tool, stock_price_tool])

# 4. Test with questions from different domains
print("\n--- Testing Medical Question ---")
question_medical = "What is hypertension?"
prediction_medical = domain_agent(question=question_medical)
print(f"Question: {question_medical}")
print(f"Agent's Answer: {prediction_medical.answer}")

print("\n--- Testing Financial Question ---")
question_financial = "What is the current stock price of AAPL?"
prediction_financial = domain_agent(question=question_financial)
print(f"Question: {question_financial}")
print(f"Agent's Answer: {prediction_financial.answer}")

print("\n--- Testing another Medical Question ---")
question_medical_2 = "Tell me about migraine."
prediction_medical_2 = domain_agent(question=question_medical_2)
print(f"Question: {question_medical_2}")
print(f"Agent's Answer: {prediction_medical_2.answer}")

# Optional: Inspect history
print("\n--- Inspecting LM History (last 3 complex ReAct trajectories) ---")
dspy.inspect_history(n=3)
