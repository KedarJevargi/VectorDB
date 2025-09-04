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

# Split text into chunks
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Main code
client = chromadb.Client()
ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text:v1.5")

# Create collection
collection = client.create_collection(name="docs")

# Load PDF
pdf_text = extract_text_from_pdf("resources/syllabus.pdf")
chunks = split_text(pdf_text, chunk_size=500, overlap=50)

# Add to ChromaDB
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# Query
results = collection.query(
    query_texts=["what are the project work activities?"],
    n_results=10
)

# Print results
for doc in results["documents"][0]:
    print(doc[:200])
    print("---")