import re
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="phi:latest",
    temperature=0
    )

messages = [
    (
        "system",
        "You are a helpful assistant that writes test cases for any programming languages like javascript, reactjs, python, java etc.",
    ),
    ("human", "generate test case code for The shopping cart should calculate total price with tax,apply discount codes, and prevent negative quantities"),
]
#  try:
ai_msg = llm.invoke(messages)
code = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
        
# Clean up markdown code blocks
code = re.sub(r'```(?:javascript|js|typescript|ts)?\n?', '', code)
code = code.strip()

print(code)
# except Exception as e:
    # print(f"Generation error: {e}")
