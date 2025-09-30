from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict
import asyncio
import os
from sentence_transformers import SentenceTransformer
from config.settings import settings
from models.document_processor import DocumentProcessor
from models.rag_engine import RAGEngine

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

print("Loading document processor...")
doc_processor = DocumentProcessor(embed_model)

print("Loading documents...")
doc_processor.load_documents("data/pdf")

print("Building indices...")
doc_processor.build_indices()

print("Loading RAG engine...")
rag_engine = RAGEngine(doc_processor)

# Models
class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 512
    temperature: float = 0.1

class ChatMessage(BaseModel):
    message: str
    sender: str

# Routes
@app.get("/")
async def root():
    return {"message": "RAG System API - Ministère de la Fonction Publique du Sénégal"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Synchronous query endpoint"""
    try:
        # This is a simplified version - in production, you'd want to handle this differently
        # For streaming, use the WebSocket endpoint
        response_parts = []
        async for token in rag_engine.generate_stream(
            request.question, 
            request.max_tokens, 
            request.temperature
        ):
            response_parts.append(token)
        
        return {"response": "".join(response_parts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")
            
            if not question:
                await websocket.send_json({"error": "Question is required"})
                continue
            
            # Send response tokens as they arrive
            async for token in rag_engine.generate_stream(question):
                await websocket.send_json({
                    "type": "token",
                    "data": token
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )