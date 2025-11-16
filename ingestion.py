# ingestion.py
import json
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# --- 1. Path to your JSON knowledge base ---
json_file_path = "D:\\ai bot\\women_legal_rights_knowledgebase_full.json"

# --- 2. Metadata extraction function ---
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["category"] = record.get("category")
    metadata["title"] = record.get("title")
    metadata["description"] = record.get("description")
    metadata["history"] = record.get("history")
    metadata["purpose"] = record.get("purpose")
    metadata["case_examples"] = record.get("case_examples")
    return metadata

# --- 3. Load JSON documents ---
loader = JSONLoader(
    file_path=json_file_path,
    jq_schema=".[]",
    content_key="description",
    metadata_func=metadata_func
)
data = loader.load()

# --- 4. Split documents into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)

# --- 5. Filter out complex metadata ---
filtered_splits = filter_complex_metadata(all_splits)

# --- 6. Initialize Ollama embeddings ---
embeddings = OllamaEmbeddings(model="mistral")  # Make sure Ollama is running

# --- 7. Create Chroma vector store and persist ---
vectorstore = Chroma.from_documents(
    documents=filtered_splits,
    embedding=embeddings,
    persist_directory="D:\\ai bot\\chroma_db"
)
vectorstore.persist()

print("✅ Data successfully loaded and stored in ChromaDB!")
