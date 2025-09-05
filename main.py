import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Split text into chunks with overlap
def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Main code
client = chromadb.Client()
ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text:v1.5")

# Create collection with Ollama embeddings
collection = client.get_or_create_collection(name="docs", embedding_function=ollama_ef)

# Load PDF
pdf_text = extract_text_from_pdf("idea.pdf")
chunks = split_text(pdf_text)

# Add to ChromaDB (will automatically use Ollama embeddings)
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# Query
results = collection.query(
    query_texts=["What is the main idea of the document?"],
    n_results=1
)

# Print results
# for doc in results["documents"][0]:
#     print(doc[:2000])
#     print("---")


print(results["distances"])