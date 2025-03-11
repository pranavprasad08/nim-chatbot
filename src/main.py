from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from pdf_processor import PDFProcessor
from chain_server import ChainServer
from vector_db import VectorDatabase
from process_images import ImageProcessor
from create_chunks import Chunker

# Initialize FastAPI
app = FastAPI()

# Define the Query Model for FastAPI
class QueryRequest(BaseModel):
    question: str

# Define Ports for Services
base_url_nim = "http://127.0.0.1:8000/v1"  # NVIDIA LLM API
base_url_embedding = "http://127.0.0.1:800l/v1"  # NVIDIA Embeddings API

# Initialize Components
vector_db = VectorDatabase()
image_processor = ImageProcessor(base_url_nim=base_url_nim)
chunker = Chunker()
pdf_processor = PDFProcessor(image_processor, chunker, vector_db)
chain_server = ChainServer(vector_db, base_url_nim=base_url_nim)  # ✅ Uses LangChain Agent



@app.post("/upload/")
async def upload_pdf(file: UploadFile):
    """Handles PDF uploads and processing."""
    pdf_processor.process_pdf(file.filename)
    return {"message": "✅ PDF processed successfully"}

@app.post("/query/")
async def query_rag(request: QueryRequest):
    """Handles user queries for RAG processing."""
    response = chain_server.query(request.question)
    return {"answer": response['output']}
