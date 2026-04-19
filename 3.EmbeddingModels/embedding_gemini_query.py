from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Using Google's embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=32
)

# Test the embeddings
text = "Hello, how are you?"
embedding = embeddings.embed_query(text)
print(f"Embedding dimension: {len(embedding)}")
print(f"Embedding: {embedding}")
