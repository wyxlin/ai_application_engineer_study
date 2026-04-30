from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a testing assistant, you are testing the API of ChatGPT.",
    input="return: API test passed.",
    max_output_tokens=30,
)

print(response.output_text)
