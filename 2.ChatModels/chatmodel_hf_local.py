from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

# Using a free open-source model from Hugging Face
# Options: 
# - "google/flan-t5-base" (good for general tasks)
# - "microsoft/DialoGPT-medium" (good for conversations)
# - "EleutherAI/gpt-neo-125M" (good for text generation)

# Try with a smaller, free model first
# This method will download all the llm files from hugging face and run it in your local machine as server
llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

# Test the LLM
model = ChatHuggingFace(llm=llm)
output = model.invoke("What is the capital of France?")
print(output.content)

