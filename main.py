import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


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


client = chromadb.Client()
ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text:v1.5")


collection = client.get_or_create_collection(
    name="docs", 
    embedding_function=ollama_ef
)



pdf_text = extract_text_from_pdf("resources/syllabus.pdf")
chunks = split_text(pdf_text)


collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)



user_query=input("Enter the query: ")


results = collection.query(
    query_texts=[user_query],
    n_results=10
)



# data = ""

for doc in results["documents"][0]:
   print(doc[:5000])
   print("-----")

   print(collection.configuration_json)
    



# import ollama

# response = ollama.chat(
#     model="mistral:7b",
#     messages=[
#         {"role": "system", "content": 
#          "You are an AI assistant that answers questions using the context provided. "
#          "If the answer is not in the context, say you dont know. "
#          "Do not make up information."
#         },
#         {"role": "user", "content": 
#          f"Context:\n{data}\n\nQuestion: {user_query}"
#         }
#     ],
    
# )

# print(response['message']['content'])


