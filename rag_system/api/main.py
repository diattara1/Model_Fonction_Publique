import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from models.vectorizer import Vectorizer
from models.generator import Generator
from agents.rag_agent import RAGAgent
from config.settings import settings

# --- Init FastAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Init RAG ---
vectorizer = Vectorizer()
generator = Generator()
agent = RAGAgent(vectorizer, generator)


# === Endpoint classique (non-streaming) ===
class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query(request: QueryRequest):
    """Réponse synchrone (pas de streaming)."""
    chunks = []
    async for token in agent.answer(request.question, stream=False):
        chunks.append(token)
    return {
        "question": request.question,
        "response": "".join(chunks).strip()
    }


# === WebSocket streaming ===
@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            question = data.strip()
            if not question:
                await websocket.send_text("⚠️ Question vide")
                continue

            # --- notifier recherche ---
            await websocket.send_text("📚 Analyse de la question...")

            # --- exécuter l’agent en streaming ---
            async for token in agent.answer(question, stream=True):
                await websocket.send_text(token)

            await websocket.send_text("\n\n✅ Fin de réponse")
    except WebSocketDisconnect:
        print("🔌 Client déconnecté")
    except Exception as e:
        print(f"❌ Erreur WebSocket: {e}")
        await websocket.send_text(f"Erreur: {str(e)}")


# === Lancement ===
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
