from fastapi import FastAPI, Request, Header
from pydantic import BaseModel
import requests
import pdfplumber
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# health check for server working 
@app.get("/")
def root():
    return {"message": "Server is working!"}


# Azure OpenAI credentials from environment
endpoint = os.getenv("OPENAI_API_BASE")
deployment = os.getenv("OPENAI_DEPLOYMENT")
subscription_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: QueryRequest, authorization: str = Header(...)):
    pdf_text = extract_text_from_pdf(request.documents)
    answers = [ask_gpt(question, pdf_text) for question in request.questions]
    return {"answers": answers}

def extract_text_from_pdf(pdf_url):
    response = requests.get(pdf_url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    text = ""
    with pdfplumber.open("temp.pdf") as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def ask_gpt(question, context):
    prompt = f"Answer this question based on the following policy:\n\n{context}\n\nQ: {question}\nA:"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=250,
        top_p=1.0,
        model=deployment
    )

    return response.choices[0].message.content.strip()




