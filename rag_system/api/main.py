#api/main.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import asyncio
import os
from sentence_transformers import SentenceTransformer
from config.settings import settings
from models.vectorizer import Vectorizer
from models.generator import Generator

# Initialize FastAPI app
app = FastAPI(title="RAG System - Ministère de la Fonction Publique du Sénégal")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
print("Loading embedding model...")
embed_model = SentenceTransformer(settings.EMBED_MODEL_NAME).to("cuda")

print("Initializing vectorizer...")
vectorizer = Vectorizer()

# Check if indices exist and load them, otherwise build them
indices_path = settings.INDICES_DIR
if os.path.exists(os.path.join(indices_path, "faiss.index")):
    print("Loading existing indices...")
    vectorizer.load_indices(indices_path)
else:
    print("Building indices from PDF directory...")
    vectorizer.load_documents(settings.PDF_DIR)
    vectorizer.build_indices()
    vectorizer.save_indices(indices_path)

print("Initializing generator...")
generator = Generator(vectorizer)

# Models
class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 512
    temperature: float = 0.1

# Routes
@app.get("/")
async def root():
    return {"message": "RAG System API - Ministère de la Fonction Publique du Sénégal"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Streaming query endpoint"""
    async def token_generator():
        try:
            async for token in generator.generate_stream(
                request.question,
                request.max_tokens,
                request.temperature
            ):
                yield token
        except Exception as e:
            yield f"[ERROR] {str(e)}"
    
    return StreamingResponse(token_generator(), media_type="text/plain")

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")
            
            if not question:
                await websocket.send_json({"error": "Question is required"})
                continue
            
            # Send response tokens as they arrive
            async for token in generator.generate_stream(question):
                await websocket.send_json({
                    "type": "token",
                    "data": token
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a new PDF document"""
    try:
        # Save uploaded file
        file_path = os.path.join(settings.PDF_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Add document to vectorizer
        vectorizer.add_document(file_path)
        
        # Save updated indices
        vectorizer.save_indices(settings.INDICES_DIR)
        
        return {"message": f"PDF {file.filename} uploaded and indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/documents")
async def list_documents():
    """List all indexed documents"""
    unique_sources = set()
    for chunk in vectorizer.chunks:
        unique_sources.add(chunk["source"])
    
    return {"documents": list(unique_sources)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )