from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Using OpenAI's embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32
)

# Test the embeddings
text = "Hello, how are you?"
embedding = embeddings.embed_query(text)
print(f"Embedding dimension: {len(embedding)}")
print(f"Embedding: {embedding}")
