from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chat = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.7, max_tokens=100)
output = chat.invoke("What is the capital of France?")
print(output)
