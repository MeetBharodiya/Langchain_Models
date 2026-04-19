from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.5)
output = chat.invoke("Write poem on ocean")
print(output.content)
