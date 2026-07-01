"""Minimal test: ask the model to continue partial code, get n diverse completions."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

partial_code = 'result = web_search(query='

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a code completion engine. When given partial Python code, output ONLY the completion — no explanation, no markdown."},
        {"role": "user", "content": f"The task is: find the best restaurants in Paris.\n\nComplete this partial code from exactly where it cuts off:\n\n{partial_code}"},
    ],
    stop=[")", "\n"],
    n=3,
    temperature=1.5,
)

print(f"Number of choices: {len(response.choices)}\n")
for i, choice in enumerate(response.choices):
    content = choice.message.content or ""
    print(f"[choice {i}] full call: {partial_code}{content})")
