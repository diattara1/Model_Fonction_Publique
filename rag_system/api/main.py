import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from models.vectorizer import Vectorizer
from models.generator import Generator
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

# Charger les indices pré-construits ou les reconstruire si besoin
try:
    vectorizer.load_indices(settings.INDICES_DIR)
    print("✅ Indices chargés avec succès")
except Exception as e:
    print(f"⚠️ Impossible de charger les indices : {e}")
    print("👉 Construction des indices depuis les PDF...")
    vectorizer.load_documents(settings.PDF_DIR)
    vectorizer.build_indices()
    vectorizer.save_indices(settings.INDICES_DIR)

generator = Generator(vectorizer)


# === WebSocket streaming ===
@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            question = (await websocket.receive_text()).strip()
            if not question:
                await websocket.send_text("⚠️ Question vide")
                continue

            # --- notifier recherche ---
            await websocket.send_text("📚 Recherche en cours...")

            # --- exécution agent en streaming ---
            async for token in generator.generate_stream(question):
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
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
