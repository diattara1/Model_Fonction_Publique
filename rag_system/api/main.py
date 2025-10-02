# api/main.py
import uvicorn
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from api.database import init_db
from api.routes import router

from config.settings import settings
from models.generator import Generator
from models.vectorizer import Vectorizer
from agents.rag_agent import RAGAgent

app = FastAPI()
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Vectorizer & Generator
vectorizer = Vectorizer()
try:
    vectorizer.load_indices(settings.INDICES_DIR)
    print("✅ Indices chargés avec succès")
except Exception as e:
    print(f"⚠️ Indices non chargés : {e}")
    vectorizer.load_documents(settings.PDF_DIR)
    vectorizer.build_indices()
    vectorizer.save_indices(settings.INDICES_DIR)

generator = Generator()
agent = RAGAgent(vectorizer, generator)

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket pour génération streaming avec mémoire conversationnelle.
    URL: ws://.../ws/generate?conv_id=123
    """
    conv_id = int(websocket.query_params.get("conv_id", "0"))
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            async for token in agent.answer(raw, conv_id=conv_id, stream=True):
                await websocket.send_text(token)
    except WebSocketDisconnect:
        # NE RIEN ÉCRIRE en DB à la déconnexion → aucun [[SESSION_BREAK]] ne fuitera
        print(f"🔌 Client déconnecté (conv_id={conv_id})")
    except Exception as e:
        print(f"❌ Erreur WebSocket: {e}")
        try:
            await websocket.send_text(f"Erreur: {str(e)}")
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
