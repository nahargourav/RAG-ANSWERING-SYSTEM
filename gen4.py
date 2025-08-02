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
        cache_start = time.time()
        print(f"Cache hit for document: {doc_url}")
        chunks, faiss_index = document_cache[doc_url]
        cache_end = time.time()
        print(f"--- Step 0: Cache retrieval complete. Duration: {cache_end - cache_start:.2f}s ---")
    else:
        print(f"Cache miss. Starting document processing for: {doc_url}")
        pdf_start = time.time()
        text = await extract_text_from_pdf_fast(doc_url)
        pdf_end = time.time()
        print(f"--- Step 1: Text extraction complete. Duration: {pdf_end - pdf_start:.2f}s ---")

        chunk_start = time.time()
        chunks = chunk_text(text)
        chunk_end = time.time()
        print(f"--- Step 2: Chunking complete. Duration: {chunk_end - chunk_start:.2f}s ---")

        embed_start = time.time()
        chunk_embeddings = await get_embeddings(chunks, model=embedding_deployment)
        embed_end = time.time()
        print(f"--- Step 3: Embedding complete. Duration: {embed_end - embed_start:.2f}s ---")

        faiss_start = time.time()
        embedding_dimension = chunk_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        faiss_index.add(chunk_embeddings)
        faiss_end = time.time()
        print(f"--- Step 4: FAISS indexing complete. Duration: {faiss_end - faiss_start:.2f}s ---")

        document_cache[doc_url] = (chunks, faiss_index)
        print(f"Document processed and cached. FAISS index created with {len(chunks)} chunks.")

    qa_start = time.time()
    # Only set cache timings if cache_start/cache_end are defined
    cache_duration = cache_end - cache_start if 'cache_start' in locals() and 'cache_end' in locals() else None
    pdf_extraction_duration = pdf_end - pdf_start if 'pdf_start' in locals() and 'pdf_end' in locals() else None
    chunking_duration = chunk_end - chunk_start if 'chunk_start' in locals() and 'chunk_end' in locals() else None
    embedding_duration = embed_end - embed_start if 'embed_start' in locals() and 'embed_end' in locals() else None
    faiss_indexing_duration = faiss_end - faiss_start if 'faiss_start' in locals() and 'faiss_end' in locals() else None
    timings = {
        "cache": cache_duration,
        "pdf_extraction": pdf_extraction_duration,
        "chunking": chunking_duration,
        "embedding": embedding_duration,
        "faiss_indexing": faiss_indexing_duration,
        "qa_batch": None,
        "total": None,
        "questions": []
    }
    question_timings = []
    async def timed_answer(q):
        q_times = {}
        q_times["question"] = q
        q_times["embedding_start"] = time.time()
        question_embedding = await get_embeddings([q], model=embedding_deployment)
        q_times["embedding_end"] = time.time()
        q_times["embedding"] = q_times["embedding_end"] - q_times["embedding_start"]
        q_times["faiss_start"] = time.time()
        relevant_chunks = search_faiss(question_embedding, faiss_index, chunks, k=5)
        q_times["faiss_end"] = time.time()
        q_times["faiss_search"] = q_times["faiss_end"] - q_times["faiss_start"]
        context = "\n---\n".join(relevant_chunks)
        q_times["gpt_start"] = time.time()
        result = await ask_gpt(q, context)
        q_times["gpt_end"] = time.time()
        q_times["gpt_answer"] = q_times["gpt_end"] - q_times["gpt_start"]
        question_timings.append(q_times)
        return result

    qa_start = time.time()
    answers = await asyncio.gather(*[timed_answer(q) for q in request.questions])
    qa_end = time.time()
    timings["qa_batch"] = qa_end - qa_start
    end_time = time.time()
    timings["total"] = end_time - start_time
    timings["questions"] = question_timings
    return {"answers": answers, "timings": timings}

# --- RAG Core Functions ---
async def answer_question_with_rag(question: str, chunks: list[str], faiss_index: faiss.Index):
    """Finds relevant context using FAISS and asks the LLM to answer, with timing analysis."""
    print(f"--- [Q] {question}")
    embed_start = time.time()
    question_embedding = await get_embeddings([question], model=embedding_deployment)
    embed_end = time.time()
    print(f"    [Q] Embedding: {embed_end - embed_start:.2f}s")

    faiss_start = time.time()
    relevant_chunks = search_faiss(question_embedding, faiss_index, chunks, k=5)
    faiss_end = time.time()
    print(f"    [Q] FAISS search: {faiss_end - faiss_start:.2f}s")

    context = "\n---\n".join(relevant_chunks)
    gpt_start = time.time()
    result = await ask_gpt(question, context)
    gpt_end = time.time()
    print(f"    [Q] GPT answer: {gpt_end - gpt_start:.2f}s")
    return result

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