import asyncio
import io
import os
import time

import faiss
import fitz  # PyMuPDF for PDF processing
import httpx
import numpy as np
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from openai import AsyncAzureOpenAI
from pydantic import BaseModel
from pymongo import MongoClient

# --- Basic Setup ---
load_dotenv()
app = FastAPI()

# --- Azure OpenAI Configuration ---
endpoint = os.getenv("OPENAI_API_BASE")
chat_deployment = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o-mini")
embedding_deployment = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")

if not all([endpoint, api_key, api_version, chat_deployment, embedding_deployment]):
    raise ValueError("Missing one or more required Azure OpenAI environment variables.")

client = AsyncAzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key, max_retries=5)

# --- MongoDB Setup ---
mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://gouravnahar3008:fM5BY3RIa0OUAifl@cluster0.junmus8.mongodb.net/")
mongo_client = MongoClient(mongo_uri)
try:
    mongo_client.admin.command('ping')
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print("❌ Connection failed:", e)
db = mongo_client["hackrx_logs"]
collection = db["requests"]

# --- Pydantic Models & Cache ---
class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

document_cache = {}

# --- API Endpoint with Logging ---
@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: QueryRequest, authorization: str = Header(None)):
    start_time = time.time()
    print(f"\n--- Request received at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    doc_url = request.documents

    if doc_url in document_cache:
        print(f"Cache hit for document: {doc_url}")
        chunks, faiss_index = document_cache[doc_url]
    else:
        print(f"Cache miss. Starting document processing for: {doc_url}")

        # Step 1: Extract text from PDF
        text_extraction_start = time.time()
        text = await extract_text_from_pdf_fast(doc_url)
        text_extraction_end = time.time()
        print(f"--- Step 1: Text extraction complete. Duration: {text_extraction_end - text_extraction_start:.2f}s ---")

        # Steps 2, 3, & 4: Chunking, Embedding, and Indexing
        indexing_start = time.time()
        chunks = chunk_text(text)
        chunk_embeddings = await get_embeddings(chunks, model=embedding_deployment)

        embedding_dimension = chunk_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        faiss_index.add(chunk_embeddings)
        indexing_end = time.time()

        document_cache[doc_url] = (chunks, faiss_index)
        print(f"--- Step 2: Indexing complete. Duration: {indexing_end - indexing_start:.2f}s ---")

    # Step 5: Answer all questions concurrently
    qa_start = time.time()
    tasks = [answer_question_with_rag(q, chunks, faiss_index, doc_url) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    qa_end = time.time()
    print(f"--- Step 3: Answering questions complete. Duration: {qa_end - qa_start:.2f}s ---")

    end_time = time.time()
    print(f"--- Total request finished. Duration: {end_time - start_time:.2f}s ---")

    return {"answers": answers}

# --- Core RAG Pipeline ---
async def answer_question_with_rag(question: str, chunks: list[str], faiss_index: faiss.Index, doc_url: str) -> str:
    question_embedding = await get_embeddings([question], model=embedding_deployment)
    relevant_chunks = search_faiss(question_embedding, faiss_index, chunks, k=5)
    context = "\n---\n".join(relevant_chunks)
    answer = await ask_gpt(question, context)

    # Log to MongoDB
    log_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "document_url": doc_url,
        "question": question,
        "answer": answer,
        "top_chunks": relevant_chunks
    }
    collection.insert_one(log_entry)

    return answer

async def ask_gpt(question: str, context: str) -> str:
    prompt = f"""
    Answer the following question based *only* on the provided context.
    If the answer is not in the context, state that the information is not available.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        model=chat_deployment
    )
    return response.choices[0].message.content.strip()

# --- Utility Functions ---
async def extract_text_from_pdf_fast(pdf_url: str) -> str:
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(pdf_url, timeout=30.0)
        response.raise_for_status()
    pdf_file_stream = io.BytesIO(response.content)

    def extract_text():
        text = ""
        with fitz.open(stream=pdf_file_stream, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, extract_text)

def chunk_text(text: str, max_tokens: int = 1000) -> list[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    paragraphs = text.split('\n\n')
    final_chunks = []
    for p in paragraphs:
        if len(tokenizer.encode(p)) > max_tokens:
            sentences = p.split('. ')
            current_chunk = ""
            for s in sentences:
                if len(tokenizer.encode(current_chunk + s + '. ')) > max_tokens:
                    final_chunks.append(current_chunk)
                    current_chunk = s + '. '
                else:
                    current_chunk += s + '. '
            final_chunks.append(current_chunk)
        else:
            final_chunks.append(p)
    return [chunk.strip() for chunk in final_chunks if chunk.strip()]

async def get_embeddings(texts: list[str], model: str, batch_size: int = 16) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
    return np.array(all_embeddings, dtype=np.float32)

def search_faiss(query_embedding: np.ndarray, index: faiss.Index, chunks: list[str], k: int = 5):
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]
