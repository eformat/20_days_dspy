import os

import dspy

# Ensure LM is configured
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)
dspy.configure(lm=lm)

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
user_msg_1 = "Hi there, my name is Bill! What can you do?"
response_1 = assistant(user_message=user_msg_1)
print(f"User: {user_msg_1}")
print(f"Assistant: {response_1.assistant_response}")

# Turn 2, referring to previous context
user_msg_2 = "What is my name?"
response_2 = assistant(user_message=user_msg_2)
print(f"\nUser: {user_msg_2}")
print(f"Assistant: {response_2.assistant_response}")


# Optional: Inspect history (shows all turns for each LM call)
print("\n--- Inspecting LM History (last 3 calls) ---")
dspy.inspect_history(n=2)
