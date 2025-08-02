# main.py

import os
import uuid
import json
import requests
import tempfile
import hashlib
import math
from fastapi import FastAPI, Depends, HTTPException, Security, APIRouter, Query
from fastapi.middleware.gzip import GZipMiddleware # <<< NEW IMPORT for Compression
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import necessary loaders and modules
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document

# --- Initial Setup & Global Objects ---
load_dotenv()

CACHE_DIR = "vectorstore_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

print("Loading models... This may take a moment.")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-4o"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={"HTTP-Referer": "http://localhost"}
)
print("✅ Models loaded successfully.")



# --- RAG Logic ---
def get_or_create_vectorstore(doc_url: str) -> FAISS:
    """
    Handles caching and processing for multiple document types.
    """
    url_hash = hashlib.md5(doc_url.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, url_hash)

    if os.path.exists(cache_path):
        print(f"Loading cached index for {doc_url}")
        return FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
    
    print(f"No cache found. Processing {doc_url}")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        try:
            response = requests.get(doc_url)
            response.raise_for_status()
            temp_file.write(response.content)
            temp_file.close()

            file_extension = os.path.splitext(doc_url.split('?')[0])[1].lower()
            if file_extension == '.pdf':
                loader = PyMuPDFLoader(temp_file.name)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(temp_file.name)
            elif file_extension == '.eml':
                loader = UnstructuredEmailLoader(temp_file.name)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(cache_path)
            print(f"Saved new index to cache for {doc_url}")
            return vectorstore
        finally:
            os.unlink(temp_file.name)

# --- API Endpoints ---
app = FastAPI(title="HackRx RAG API (Final Optimized Version)")
app.add_middleware(GZipMiddleware, minimum_size=1000) # <<< NEW: Add compression middleware

router = APIRouter(prefix="/api/v1")
security = HTTPBearer()
API_KEY = os.getenv("MY_API_KEY")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

# <<< NEW: Response model now includes pagination metadata >>>
class PaginatedHackRxResponse(BaseModel):
    answers: List[str]

@router.post("/hackrx/run", response_model=PaginatedHackRxResponse)
def run_hackrx_job(
    request: HackRxRequest,
    is_authenticated: bool = Depends(verify_api_key),
    # <<< NEW: Pagination query parameters >>>
    page: int = Query(1, ge=1, description="Page number to retrieve"),
    page_size: int = Query(10, ge=1, le=100, description="Number of questions per page")
):
    """
    Accepts a document URL and a list of questions, processes them synchronously
    with pagination, and returns the final answers for the requested page.
    """
    try:
        total_questions = len(request.questions)
        total_pages = math.ceil(total_questions / page_size)

        # Calculate the slice of questions for the current page
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        questions_for_page = request.questions[start_index:end_index]

        if not questions_for_page:
            raise HTTPException(status_code=404, detail="Page number out of range. No questions on this page.")

        vectorstore = get_or_create_vectorstore(str(request.documents))
        retriever = vectorstore.as_retriever()

        prompt_template = """
        ## Role
You are a highly specialized Financial Expert and Retrieval-Augmented Generation (RAG) AI Assistant with deep expertise in financial document analysis and precise information extraction. Your core mission is to provide authoritative, accurate, and concise financial insights directly sourced from provided documents.

## Task
Analyze financial documents and answer specific questions with utmost precision, acting as a sophisticated financial intelligence system that extracts and communicates critical financial information with expert-level accuracy.

## Context
As a financial knowledge navigator, your purpose is to bridge complex financial documentation with clear, direct answers. You serve professionals, researchers, and decision-makers who require immediate, reliable financial insights without unnecessary elaboration.

## Instructions
1. Document Analysis Protocol:
- Treat each financial document as a sacred source of truth
- Read the entire context meticulously before answering
- Focus exclusively on information present in the provided document

2. Answer Generation Rules:
- Provide answers that are:
  * 100% factual
  * Directly sourced from the context
  * Concise and information-dense
  * Free from personal interpretation or external knowledge

3. Response Constraints:
- If the answer exists in the document: Deliver a single, compact paragraph
- If NO answer is found: Respond EXACTLY with "The provided document does not contain an answer to this question."
- Eliminate all conversational elements, introductions, or supplementary commentary

4. Expertise Persona:
- Embody a professional financial analyst
- Demonstrate unwavering commitment to accuracy
- Treat each query as a critical financial intelligence mission

5. Error Handling:
- Never fabricate information
- Do not supplement answers with external data
- Maintain strict adherence to the source document's content

6. Operational Mandate:
- Your primary directive is extracting precise financial knowledge
- Prioritize clarity, brevity, and factual integrity above all else

CRITICAL DIRECTIVE: Your professional reputation and the financial decisions dependent on your insights demand absolute precision and truthfulness.
Context:
{context}

Questions:
{question}
        """
        RAG_PROMPT = PromptTemplate.from_template(prompt_template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )

        print(f"Answering {len(questions_for_page)} questions for page {page}...")
        answers = rag_chain.batch(questions_for_page, {"max_concurrency": 10})
        print("✅ Page processed.")
        
        return PaginatedHackRxResponse(
            answers=answers,
            
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

app.include_router(router)
