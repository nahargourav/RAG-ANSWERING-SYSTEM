import asyncio
import io
import os
import time

import fitz
import httpx
import numpy as np
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from openai import AsyncAzureOpenAI
from pydantic import BaseModel
from pinecone import Pinecone

# --- Basic Setup ---
load_dotenv()
app = FastAPI()

# --- Azure OpenAI Configuration (for answer generation only) ---
endpoint = os.getenv("OPENAI_API_BASE")
chat_deployment = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o-mini")
api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")

if not all([endpoint, api_key, api_version, chat_deployment]):
    raise ValueValue("Missing one or more required Azure OpenAI environment variables.")

client = AsyncAzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key, max_retries=5)

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not all([PINECONE_API_KEY, PINECONE_INDEX]):
    raise ValueError("Missing one or more required Pinecone environment variables.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# --- Pydantic Models & Cache ---
class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

document_cache = {}

# --- API Endpoint with Detailed Timing ---
@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: QueryRequest, authorization: str = Header(None)):
    start_time = time.time()
    doc_url = request.documents
    namespace = doc_url.replace('https://', '').replace('/', '_')  # Simple namespace from URL

    if namespace in document_cache:
        chunks = document_cache[namespace]
    else:
        # Extract text from PDF
        text = await extract_text_from_pdf_fast(doc_url)
        chunks = chunk_text(text)
        # Use Pinecone Inference API for embeddings
        chunk_embeddings = await pinecone_inference_embed(chunks)
        await upsert_to_pinecone(namespace, chunks, chunk_embeddings)
        document_cache[namespace] = chunks

    # Answer questions
    answers = await asyncio.gather(*[answer_question_with_pinecone(q, namespace, chunks) for q in request.questions])
    end_time = time.time()

    return {"answers": answers, }

# --- RAG Core Functions ---
async def answer_question_with_pinecone(question: str, namespace: str, chunks: list[str]):
    question_embedding = await pinecone_inference_embed([question])
    relevant_chunks = await search_pinecone(question_embedding[0], namespace)
    context = "\n---\n".join(relevant_chunks)
    return await ask_gpt(question, context)

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

def chunk_text(text: str, max_tokens: int = 1800) -> list[str]:
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

async def pinecone_inference_embed(texts: list[str]) -> np.ndarray:
    embeddings = pc.inference.embed(
        "llama-text-embed-v2",
        inputs=texts,
        parameters={
            "input_type": "passage" if len(texts) > 1 else "query",  # 'passage' for documents, 'query' for single questions
            "dimension": 1024,  # Match your index dimension
            "truncate": "END"
        }
    )
    return np.array([emb.values for emb in embeddings], dtype=np.float32)

async def upsert_to_pinecone(namespace: str, chunks: list[str], embeddings: np.ndarray):
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": emb.tolist(),
            "metadata": {"text": chunk}
        })
    index.upsert(vectors=vectors, namespace=namespace)

async def search_pinecone(query_embedding: np.ndarray, namespace: str, k: int = 5):
    response = index.query(
        vector=query_embedding.tolist(),
        top_k=k,
        namespace=namespace,
        include_metadata=True
    )
    return [match['metadata']['text'] for match in response['matches']]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
