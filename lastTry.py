import asyncio
import io
import os
import time

import faiss
import fitz  # PyMuPDF
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
embedding_deployment = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")

client = AsyncAzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key, max_retries=5)

# --- MongoDB Setup ---
mongo_uri = os.getenv("MONGO_URI")
mongo_client = MongoClient(mongo_uri)
mongo_client.admin.command('ping')
db = mongo_client["hackrx_logs"]
collection = db["CheckRequest"]

# --- Pydantic Model ---
class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

document_cache = {}

# --- Main Endpoint ---
@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: QueryRequest, authorization: str = Header(None)):
    start_time = time.time()
    doc_url = request.documents

    if doc_url in document_cache:
        chunks, faiss_index = document_cache[doc_url]
    else:
        text = await extract_text_from_pdf_fast(doc_url)
        chunks = smart_chunk_text(text)
        chunk_embeddings = await get_embeddings(chunks, model=embedding_deployment)

        embedding_dim = chunk_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(chunk_embeddings)

        document_cache[doc_url] = (chunks, faiss_index)

    tasks = [answer_question_with_rag(q, chunks, faiss_index, doc_url) for q in request.questions]
    answers = await asyncio.gather(*tasks)

    return {"answers": answers}

# --- Semantic Expansion ---
def expand_question_semantics(question: str) -> list[str]:
    synonyms = {
        "IVF": ["in vitro fertilization", "assisted reproduction", "ART", "infertility treatment"],
        "settled": ["paid", "reimbursed", "processed"],
        "hospitalization": ["hospital admission", "inpatient care"],
    }
    expanded = [question]
    for term, alts in synonyms.items():
        if term.lower() in question.lower():
            for alt in alts:
                expanded.append(question.replace(term, alt))
    return list(set(expanded))

# --- Answer Pipeline ---
async def answer_question_with_rag(question: str, chunks: list[str], faiss_index: faiss.Index, doc_url: str) -> str:
    expanded_questions = expand_question_semantics(question)
    question_embeddings = await get_embeddings(expanded_questions, model=embedding_deployment)
    avg_embedding = np.mean(question_embeddings, axis=0, keepdims=True)

    retrieved_chunks = search_faiss(avg_embedding, faiss_index, chunks, k=10)
    top_chunks = rerank_chunks_by_keyword_overlap(question, retrieved_chunks, top_k=5)
    context = "\n---\n".join(top_chunks)
    answer = await ask_gpt(question, context)

    collection.insert_one({
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "document_url": doc_url,
        "question": question,
        "answer": answer,
        "top_chunks": top_chunks
    })

    return answer

def rerank_chunks_by_keyword_overlap(question: str, chunks: list[str], top_k: int = 5) -> list[str]:
    q_words = set(question.lower().split())
    ranked = sorted(chunks, key=lambda c: sum(w in c.lower() for w in q_words), reverse=True)
    return ranked[:top_k]

async def ask_gpt(question: str, context: str) -> str:
    prompt = f"""
You are a helpful assistant specialized in insurance policy documents.

Use the context below to answer the question. Even if the exact term from the question (e.g., 'IVF') is not present, identify semantically related terms (e.g., 'assisted reproduction').

If partial answers exist, explain them clearly. If the document explicitly excludes or limits something, state it confidently. 
Only say "Information not available in the document." if the context is truly unrelated or insufficient.

---
Context:
{context}
---
Question: {question}
Answer (concise and factual in 2–3 sentences):
"""
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        model=chat_deployment
    )
    return response.choices[0].message.content.strip()

# --- PDF Text Extraction ---
async def extract_text_from_pdf_fast(pdf_url: str) -> str:
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(pdf_url, timeout=30.0)
        response.raise_for_status()
    pdf_file_stream = io.BytesIO(response.content)

    def extract_text():
        with fitz.open(stream=pdf_file_stream, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, extract_text)

# --- Improved Chunking ---
def smart_chunk_text(text: str, max_len: int = 800) -> list[str]:
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    chunks, buffer = [], ""
    for p in paras:
        if len(buffer) + len(p) < max_len:
            buffer += " " + p
        else:
            chunks.append(buffer.strip())
            buffer = p
    if buffer:
        chunks.append(buffer.strip())
    return [c for c in chunks if len(c.split()) > 20]

# --- Embedding Generation ---
async def get_embeddings(texts: list[str], model: str, batch_size: int = 16) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
    return np.array(all_embeddings, dtype=np.float32)

# --- FAISS Search ---
def search_faiss(query_embedding: np.ndarray, index: faiss.Index, chunks: list[str], k: int = 10):
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
