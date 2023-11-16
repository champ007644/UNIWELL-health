import openai

# Set your OpenAI API key here
openai.api_key = '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'

def ask_openai(question, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=question,
        max_tokens=150
    )
    return response.choices[0].text.strip()

print("Welcome to the Health Chatbot. You can start by asking a question or stating your concern.")

while True:
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Health Chatbot: Goodbye!")
        break

    response = ask_openai(user_input, model="text-davinci-003")
    print("Health Chatbot:", response)