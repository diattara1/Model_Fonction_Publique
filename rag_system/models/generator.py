import torch
import asyncio
import os
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from typing import List, Dict, Optional, AsyncGenerator, TypedDict
from config.settings import settings
from models.vectorizer import Vectorizer
from langgraph.graph import StateGraph, END


class RAGState(TypedDict):
    question: str
    documents: List[str]
    reformulé: bool
    reformulation_count: int
    réponse: Optional[str]


class Generator:
    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL_NAME,
            trust_remote_code=True,
            token=settings.HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        # Relie tokenizer/model au vectorizer
        self.vectorizer.tokenizer = self.tokenizer
        self.vectorizer.model = self.model

        # Initialise le graphe d’agent
        self.app = self._create_agent_graph()

    # === Génération non-streaming (complète) ===
    def _llm_generate(self, prompt: str, max_new_tokens=256, temperature=0.0) -> str:
        """Génère une réponse complète (sans streaming)"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0
            )[0]

        input_length = inputs.input_ids.shape[1]
        response_ids = out_ids[input_length:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # === Graphe RAG ===
    def _create_agent_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("agent_node", self.agent_node)
        graph.set_entry_point("agent_node")

        def decision_router(state):
            return "answer" if state.get("réponse") else "terminate"

        graph.add_conditional_edges("agent_node", decision_router, {"answer": END, "terminate": END})
        return graph.compile()

    # === Outils ===
    def tool_reformulate(self, question: str, docs: list[dict]) -> str:
        context = "\n".join(f"- {d['text'][:500]}..." for d in docs[:3])
        prompt = (
            "Tu es un agent de la fonction publique du Sénégal. "
            "La question suivante n’a pas obtenu de réponse suffisante. "
            "Voici les extraits les plus pertinents trouvés :\n"
            f"{context}\n\n"
            "Reformule la question pour cibler plus précisément l'information manquante, "
            "sans changer son sens.\n"
            f"Question originale : {question}\n"
            "Nouvelle question :"
        )
        return self._llm_generate(prompt, max_new_tokens=128, temperature=0.0)

    def tool_judge(self, question: str, docs: list[dict]) -> str:
        if not docs:
            return "REFORMULER"

        context = "\n\n".join(f"- {d['text']}" for d in docs)
        prompt = f"""Sur une échelle de 1-10, à quel point ces extraits permettent-ils de répondre à la question ?
Réponds uniquement avec le chiffre.

Question: {question}

Extraits:
{context}

Score (1-10):"""

        score_text = self._llm_generate(prompt, max_new_tokens=3, temperature=0.0)
        try:
            score = int(score_text.strip())
            return "PERTINENT" if score >= 6 else "REFORMULER"
        except:
            return "REFORMULER"

    # === Génération streaming ===
    async def _stream_answer(self, question: str, docs: list[dict]) -> AsyncGenerator[str, None]:
        """Streaming token-par-token via TextIteratorStreamer"""
        context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
        prompt = (
            "Réponds à la question en 5 phrases maximum.\n"
            "Inclue 1-2 citations EXACTES (entre guillemets) tirées des extraits.\n"
            "Ajoute une ligne 'Sources: ...' listant fichier:page.\n\n"
            f"Question:\n{question}\n\nExtraits:\n{context}\n\nRéponse:"
        )

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # ⚡ streamer consommé en direct → streaming réel
        for new_text in streamer:
            yield new_text
            await asyncio.sleep(0)

        thread.join()

        # Ajout des sources
        sources = [f"{os.path.basename(d['source'])}:p.{d['page']+1}" for d in docs]
        yield f"\n\n📚 Sources: " + ", ".join(sorted(set(sources)))

    # === Agent node (logique RAG) ===
    def agent_node(self, state: RAGState):
        question = state["question"]

        for step in range(1, settings.MAX_STEPS + 1):
            docs = self.vectorizer.hybrid_search(question, top_k=settings.TOP_K)
            verdict = self.tool_judge(question, docs)

            if verdict == "PERTINENT":
                answer = self._llm_generate(
                    f"Réponds brièvement à la question : {question}",
                    max_new_tokens=256,
                    temperature=0.1
                )
                return {
                    "decision": "answer",
                    "documents": [d["text"] for d in docs],
                    "réponse": answer,
                    "question": question
                }
            else:
                if step == settings.MAX_STEPS:
                    return {
                        "decision": "terminate",
                        "documents": [d["text"] for d in docs],
                        "réponse": "Je n’ai trouvé aucun document pertinent."
                    }
                question = self.tool_reformulate(question, docs)

        return {
            "decision": "terminate",
            "documents": [],
            "réponse": "Arrêt sans réponse."
        }

    # === Interfaces ===
    async def generate_stream(self, question: str, max_tokens: int = 256, temperature: float = 0.1):
        """Streaming complet via agent_node"""
        for step in range(1, settings.MAX_STEPS + 1):
            docs = self.vectorizer.hybrid_search(question, top_k=settings.TOP_K)
            verdict = self.tool_judge(question, docs)

            if verdict == "PERTINENT":
                async for token in self._stream_answer(question, docs):
                    yield token
                return
            else:
                if step == settings.MAX_STEPS:
                    yield "\n❌ Aucun document pertinent trouvé.\n"
                    return
                question = self.tool_reformulate(question, docs)

    def generate_sync(self, question: str) -> str:
        """Non-streaming (utilise agent_node via LangGraph)"""
        initial_state = {
            "question": question,
            "documents": [],
            "reformulé": False,
            "reformulation_count": 0,
            "réponse": None
        }
        final_state = self.app.invoke(initial_state)
        return final_state.get("réponse", "Aucune réponse générée.")
