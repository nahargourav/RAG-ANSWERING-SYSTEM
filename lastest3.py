import asyncio
import io
import os
import time

import faiss
import fitz
import httpx
import numpy as np
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

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

# --- Pydantic Models & Cache ---
class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

document_cache = {}

# --- API Endpoint with Detailed Timing ---
@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: QueryRequest, authorization: str = Header(None)):
    """
    Endpoint using a highly optimized RAG pipeline with detailed timing for each step.
    """
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
        print(f"--- Step 2: Indexing (Chunking, Embedding, FAISS) complete. Duration: {indexing_end - indexing_start:.2f}s ---")
        print(f"Document processed and cached. FAISS index created with {len(chunks)} chunks.")

    # Step 5: Answer all questions concurrently
    qa_start = time.time()
    tasks = [answer_question_with_rag(q, chunks, faiss_index) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    qa_end = time.time()
    print(f"--- Step 3: Answering all questions complete. Duration: {qa_end - qa_start:.2f}s ---")
    
    end_time = time.time()
    print(f"--- Total request finished at {time.strftime('%Y-%m-%d %H:%M:%S')}. Total duration: {end_time - start_time:.2f}s ---")
    
    return {"answers": answers}

# --- RAG Core Functions ---
async def answer_question_with_rag(question: str, chunks: list[str], faiss_index: faiss.Index):
    """Finds relevant context using FAISS and asks the LLM to answer."""
    question_embedding = await get_embeddings([question], model=embedding_deployment)
    relevant_chunks = search_faiss(question_embedding, faiss_index, chunks, k=5)
    context = "\n---\n".join(relevant_chunks)
    return await ask_gpt(question, context)

async def ask_gpt(question: str, context: str) -> str:
    """Sends a question and concise context to the OpenAI model."""
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

# --- Helper Functions ---
async def extract_text_from_pdf_fast(pdf_url: str) -> str:
    """Asynchronously downloads a PDF and extracts its text using the faster PyMuPDF library."""
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
    """Splits text into chunks of a maximum size using a tokenizer."""
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
    """Generates embeddings for a list of texts in batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
    return np.array(all_embeddings, dtype=np.float32)

def search_faiss(query_embedding: np.ndarray, index: faiss.Index, chunks: list[str], k: int = 5):
    """Searches the FAISS index for the k most similar chunks."""
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]
