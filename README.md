# 🚀 RAG-Based Document & API Question Answering System

## 📌 Overview
This project is a **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **FAISS**, **Azure OpenAI**, and **MongoDB**.  
It processes PDF documents and API endpoints to answer user queries with high accuracy.  

It supports:
- 📄 **PDF document ingestion and semantic search**
- 🌐 **Custom API-based logic evaluation** (e.g., retrieving flight numbers & secret tokens)
- 🧠 **Semantic expansion** for better question understanding
- 🗄 **MongoDB logging** for requests, answers, and context

---

## ✨ Features
- **Smart PDF Text Extraction** (via `PyMuPDF`)
- **Intelligent Text Chunking** for better retrieval
- **Embeddings Generation** with Azure OpenAI (`text-embedding-3-large`)
- **FAISS Vector Search** for high-speed retrieval
- **Custom Logic** for:
  - Retrieving flight numbers based on a user's favorite city
  - Extracting secret tokens from HTML pages
- **Semantic Understanding**: interprets related terms like *IVF ≈ assisted reproduction*
- **Detailed MongoDB Logging** of:
  - Incoming request
  - Extracted context chunks
  - Final answers & API results
- **Optimized GPT Prompting** for short, keyword-rich answers

---

## 🛠 Tech Stack
- **Backend Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **LLM Provider:** [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- **Vector Store:** [FAISS](https://faiss.ai/)
- **Database:** [MongoDB Atlas](https://www.mongodb.com/atlas/database)
- **PDF Parsing:** [PyMuPDF](https://pymupdf.readthedocs.io/)
- **HTML Parsing:** [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)

---

## 📂 Project Structure

```yaml
📦 hackrx-rag-system
├── log.py               # Main FastAPI server
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
````

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/nahargourav/RAG-ANSWERING-SYSTEM.git
````

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Create `.env` File

```env
OPENAI_API_BASE=https://your-azure-openai-endpoint
OPENAI_API_KEY=your-azure-api-key
OPENAI_API_VERSION=2024-02-15-preview
OPENAI_DEPLOYMENT=gpt-4o-mini
OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net
```

---

## ▶️ Running the Server

```bash
uvicorn log:app --reload --host 0.0.0.0 --port 8000
```

Your API will be available at:
**`http://localhost:8000`**

---

## 📡 API Endpoints

### **POST** `/api/v1/hackrx/run`

#### Request Body:

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": ["What is my flight number?"]
}
```

#### Special Cases:

* If the `documents` URL contains `"get-secret-token"`, the system fetches the HTML and extracts the token.
* If the question involves flights, the system calls HackRx APIs to get the flight number.

#### Response:

```json
{
  "answers": ["Favorite city: New York, Flight Number: XYZ123"]
}
```

---

## 🧪 Example Requests

**Secret Token Retrieval**

```json
{
  "documents": "https://register.hackrx.in/utils/get-secret-token?hackTeam=5563",
  "questions": ["Go to the link and get the secret token and return it"]
}
```

**Flight Number Retrieval**

```json
{
  "documents": "https://hackrx.blob.core.windows.net/hackrx/rounds/FinalRound4SubmissionPDF.pdf",
  "questions": ["What is my flight number?"]
}
```

---

## 🗄 MongoDB Logging

Every request is logged in MongoDB:

* `timestamp`
* `auth_header`
* `request_data`
* `document_url`
* `question`
* `answer`
* `top_chunks` (retrieved document chunks)
* `api_result` (for special API logic)

---

## 📌 Notes

* Ensure **Azure OpenAI** and **MongoDB Atlas** credentials are set in `.env`
* Install system packages for PyMuPDF if running on Linux:


---

