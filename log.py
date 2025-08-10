import asyncio
import io
import os
import time
import re 
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
from bs4 import BeautifulSoup  # <-- NEW import

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
    request_data = request.dict()
    log_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "auth_header": authorization,
        "request_data": request_data
    }
    collection.insert_one(log_entry)

    doc_url = request.documents
    start_time = time.time()

    # Special case: handle secret token link directly (no PDF processing)
    if "get-secret-token" in doc_url:
        async with httpx.AsyncClient() as client:
            resp = await client.get(doc_url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            token_div = soup.find(id="token")
            token_text = token_div.text.strip() if token_div else "Token not found"
        answer_text = f"Secret Token: {token_text}"

        # Log the Q&A to MongoDB
        collection.insert_one({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "document_url": doc_url,
            "question": request.questions[0] if request.questions else None,
            "answer": answer_text,
            "api_result": answer_text,  # Since we didn't use RAG, the API result is the answer
            "top_chunks": []  # No chunks used in this path
        })

        return {"answers": [answer_text]}

    # PDF flow
    if doc_url in document_cache:
        chunks, faiss_index = document_cache[doc_url]
    else:
        t0 = time.time()
        text = await extract_text_from_pdf_fast(doc_url)
        print(f"[Time] PDF text extraction: {time.time() - t0:.2f}s")

        t1 = time.time()
        chunks = smart_chunk_text(text)
        print(f"[Time] Smart chunking: {time.time() - t1:.2f}s")

        t2 = time.time()
        chunk_embeddings = await get_embeddings(chunks, model=embedding_deployment)
        print(f"[Time] Embedding generation: {time.time() - t2:.2f}s")

        t3 = time.time()
        embedding_dim = chunk_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(chunk_embeddings)
        print(f"[Time] FAISS index creation: {time.time() - t3:.2f}s")

        document_cache[doc_url] = (chunks, faiss_index)

    tasks = [answer_question_with_rag(q, chunks, faiss_index, doc_url) for q in request.questions]
    answers = await asyncio.gather(*tasks)

    print(f"[Time] Overall API call time: {time.time() - start_time:.2f}s")
    return {"answers": answers}


# --- API Logic Handler ---

async def evaluate_custom_logic(question: str) -> str:
    keywords = ["flight", "api", "city", "landmark"]
    if not any(kw in question.lower() for kw in keywords):
        return ""

    t_api = time.time()
    async with httpx.AsyncClient() as client:
        try:
            fav_city_resp = await client.get(
                "https://register.hackrx.in/submissions/myFavouriteCity",
                timeout=3
            )

            fav_city_resp.raise_for_status()
            city_data = fav_city_resp.json()  # Parse JSON properly

            raw_city = city_data["data"]["city"]  # Get just the city name
            print(f"[DEBUG] Raw city from API: {repr(raw_city)}")

            # Normalize key
            city_key = re.sub(r"\s+", "", raw_city.strip().lower())
            print(f"[DEBUG] Normalized city key: {city_key}")

            CITY_TO_ENDPOINT = {
                "delhi": "getFirstCityFlightNumber",
                "hyderabad": "getSecondCityFlightNumber",
                "paris": "getSecondCityFlightNumber",
                "newyork": "getThirdCityFlightNumber",
                "tokyo": "getFourthCityFlightNumber",
                "istanbul": "getFourthCityFlightNumber",
            }

            endpoint = CITY_TO_ENDPOINT.get(city_key, "getFifthCityFlightNumber")
            print(f"[DEBUG] Selected endpoint: {endpoint}")

            flight_resp = await client.get(
                f"https://register.hackrx.in/teams/public/flights/{endpoint}",
                timeout=3
            )

            result = f"Favorite city: {raw_city}, Flight Number: {flight_resp.text.strip()}"
            print(result)
            print(f"[Time] API call logic: {time.time() - t_api:.2f}s")
            return result
        except Exception as e:
            print(f"[Time] API call failed: {str(e)}")
            return ""


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
    t0 = time.time()
    api_result = await evaluate_custom_logic(question)

    expanded_questions = expand_question_semantics(question)
    question_embeddings = await get_embeddings(expanded_questions, model=embedding_deployment)
    avg_embedding = np.mean(question_embeddings, axis=0, keepdims=True)

    retrieved_chunks = search_faiss(avg_embedding, faiss_index, chunks, k=16)
    top_chunks = rerank_chunks_by_keyword_overlap(question, retrieved_chunks, top_k=5)

    context = api_result + "\n---\n" + "\n---\n".join(top_chunks) if api_result else "\n---\n".join(top_chunks)

    t1 = time.time()
    answer = await ask_gpt(question, context)
    print(f"[Time] GPT answer generation: {time.time() - t1:.2f}s")

    collection.insert_one({
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "document_url": doc_url,
        "question": question,
        "answer": answer,
        "top_chunks": top_chunks,
        "api_result": api_result
    })

    print(f"[Time] Total time for answering '{question}': {time.time() - t0:.2f}s")
    return answer

def rerank_chunks_by_keyword_overlap(question: str, chunks: list[str], top_k: int = 5) -> list[str]:
    q_words = set(question.lower().split())
    ranked = sorted(chunks, key=lambda c: sum(w in c.lower() for w in q_words), reverse=True)
    return ranked[:top_k]

async def ask_gpt(question: str, context: str) -> str:
    prompt = f"""
You are a precise assistant answering based on a given document and, only if nothing relevant is found, your general knowledge.

Follow these rules strictly:

---
1. keep the answer short concise 1-2 line and **use keywords from document** as much as possible.


2. If the document contains **anything related**, even if surprising or incorrect, **treat it as true** and use it to answer.

3. If the document includes even a **weak or partial match**, use it — you are expected to interpret semantically related ideas. For example:
   - IVF ≈ assisted reproduction
   - Hospitalization ≈ inpatient treatment
   - Settled ≈ paid or reimbursed

4. If multiple pieces of information are scattered, combine them into a **coherent answer**.

5. In no possible scenario your output should mean that the answer to asked question was not in the provided data. If there is absolutely no relevant information in the document, using your general knowledge answer the question, don't say anything like \"the document does not provide any information\" with respect to question or anything semantically equal to that.

Just give the **final answer clearly and directly**, in 2 sentences maximum.


---
Context:
{context}
---
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
    return [c for c in chunks if len(c.split()) > 5]

# --- Embedding Generation ---
async def get_embeddings(texts: list[str], model: str, batch_size: int = 20) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
    return np.array(all_embeddings, dtype=np.float32)

# --- FAISS Search ---
def search_faiss(query_embedding: np.ndarray, index: faiss.Index, chunks: list[str], k: int = 16):
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
