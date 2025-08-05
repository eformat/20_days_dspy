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

print("--- Day 14: Conversational Assistant with History ---")


# 1. Define a signature for a conversational turn
class ChatTurn(dspy.Signature):
    """Engage in a helpful conversation, using previous conversation history."""

    history: dspy.History = dspy.InputField(
        desc="Previous turns of the conversation as a History object"
    )
    user_message: str = dspy.InputField()
    assistant_response: str = dspy.OutputField()


print("\n--- Building a Stateful Conversational Assistant ---")


# 2. Define a Conversational Assistant module
class ConversationalAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        self.chat = dspy.Predict(ChatTurn)
        self.conversation_history = dspy.History(messages=[])

    def forward(self, user_message: str):
        # Add the current user message to history before generating response
        current_history = dspy.History(messages=self.conversation_history.messages)

        # Call the chat module with current history and user message
        prediction = self.chat(history=current_history, user_message=user_message)

        # Add the assistant's response to history
        self.conversation_history.messages.append(
            {
                "user_message": user_message,
                "assistant_response": prediction.assistant_response,
            }
        )

        return prediction


# 3. Instantiate the assistant
assistant = ConversationalAssistant()

# 4. Simulate a conversation
print("\n--- Conversation Simulation ---")

# Turn 1
user_msg_1 = "Hi there, my name is Bill! I am 54 years old. What can you do?"
response_1 = assistant(user_message=user_msg_1)
print(f"User: {user_msg_1}")
print(f"Assistant: {response_1.assistant_response}")

# Turn 2, referring to previous context
user_msg_2 = "What is my name?"
response_2 = assistant(user_message=user_msg_2)
print(f"\nUser: {user_msg_2}")
print(f"Assistant: {response_2.assistant_response}")

# Turn 3, referring to previous context
user_msg_3 = "Based on my age, what kind of music would i like?"
response_3 = assistant(user_message=user_msg_3)
print(f"\nUser: {user_msg_3}")
print(f"Assistant: {response_3.assistant_response}")


# Optional: Inspect history (shows all turns for each LM call)
print("\n--- Inspecting LM History (last 3 calls) ---")
dspy.inspect_history(n=3)
