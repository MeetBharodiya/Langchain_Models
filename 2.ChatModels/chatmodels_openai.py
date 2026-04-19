from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model="gpt-3.4", temperature=0.7, max_completion_tokens=100)
output = chat.invoke("What is the capital of France?")
print(output)