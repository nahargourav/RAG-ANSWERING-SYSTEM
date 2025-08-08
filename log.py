from fastapi import FastAPI, Header
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import time

# --- Load environment variables ---
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# --- MongoDB setup ---
mongo_client = MongoClient(mongo_uri)
db = mongo_client["hackrx_logs"]
collection = db["LoggedRequests"]

# --- FastAPI app ---
app = FastAPI()

# --- Request model ---
class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

# --- Endpoint ---
@app.post("/api/v1/hackrx/run")
async def log_query_request(request: QueryRequest, authorization: str = Header(None)):
    # Convert request to dict (works regardless of attribute names)
    request_data = request.dict()

    # Add additional metadata
    log_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "auth_header": authorization,
        "request_data": request_data
    }

    # Insert into MongoDB
    result = collection.insert_one(log_entry)

    return {
        "status": "success",
        "inserted_id": str(result.inserted_id)
    }
