import asyncio
import io
import os

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

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: QueryRequest, authorization: str = Header(None)):
    doc_url = request.documents

    if doc_url in document_cache:
        chunks, faiss_index = document_cache[doc_url]
    else:
        text = await extract_text_from_pdf_fast(doc_url)
        chunks = chunk_text(text)
        chunk_embeddings = await get_embeddings(chunks, model=embedding_deployment)
        embedding_dimension = chunk_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        faiss_index.add(chunk_embeddings)
        document_cache[doc_url] = (chunks, faiss_index)

    async def answer(q):
        question_embedding = await get_embeddings([q], model=embedding_deployment)
        relevant_chunks = search_faiss(question_embedding, faiss_index, chunks, k=5)
        context = "\n---\n".join(relevant_chunks)
        return await ask_gpt(q, context)

    answers = await asyncio.gather(*[answer(q) for q in request.questions])
    return {"answers": answers}

# --- Updated GPT Answer Function for Concise Output ---
async def ask_gpt(question: str, context: str) -> str:
    prompt = f"""
    You are a helpful assistant. 
    Answer the following question in a short, clear, and complete way (1–2 sentences maximum).
    Use only the provided context. If the answer is not present, reply: \"Information not available in the document.\"

    Context:
    {context}

    Question: {question}
    Answer:
    """
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
        model=chat_deployment
    )
    return response.choices[0].message.content.strip()

# --- Helper Functions ---
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
