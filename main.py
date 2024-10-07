import textgrad as tg
import os
from textgrad.engine.openai import AzureChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the Chain-of-Thoughts prompt
cot_prompt = """
You are an AI assistant that uses Chain-of-Thoughts reasoning to answer questions.
Follow these steps:
1. Restate the question
2. Break down the problem into steps
3. Solve each step
4. Combine the results to get the final answer
5. State the final answer clearly

Provide your reasoning for each step.
"""

# Set up the Azure OpenAI engine with the system prompt
engine = AzureChatOpenAI(
    model_string="gpt4o",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    system_prompt=cot_prompt  # Set the system prompt here
)

# Set the backward engine
tg.set_backward_engine(engine)

# Create the query
query = tg.Variable(
    "How many r's there in Strawberrry?",
    requires_grad=False,
    role_description="question to be answered"
)

# Create the model
model = tg.BlackboxLLM(engine)

# Generate a response
response = model(query)  # Remove system_prompt from here

# Print the response
print(response.value)