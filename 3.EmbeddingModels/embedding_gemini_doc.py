from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Using Google's embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=32
)

# Test the embeddings
# text = "Hello, how are you?"
documents = [
    "Hello, how are you?",
    "I am fine, how about you?",
    "I am also fine, thanks for asking.",
]
embedding = embeddings.embed_documents(documents)
print(f"Number of documents: {len(embedding)}")
print(f"Embedding dimension per document: {len(embedding[0])}")
print(f"Embeddings: {embedding}")
