from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

# text = "Hello, how are you?"
embedding = embeddings.embed_query(text)

print(len(embedding))
print(embedding)

# Test multiple document embeddings
docs = [
    "LangChain helps build AI applications",
    "Embeddings convert text into vectors",
    "Hugging Face provides open-source models",
    "Semantic search finds similar meaning"
]

doc_embeddings = embeddings.embed_documents(docs)

print("\nDocument Embeddings Test:")
print(f"Total documents: {len(doc_embeddings)}")
print(f"Embedding dimension: {len(doc_embeddings[0])}")

# Print first document sample
print("\nFirst document embedding (first 5 values):")
print(doc_embeddings)