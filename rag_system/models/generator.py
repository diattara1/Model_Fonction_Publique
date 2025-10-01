import torch
import asyncio
import time
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
        # Set tokenizer and model in vectorizer for search
        self.vectorizer.tokenizer = self.tokenizer
        self.vectorizer.model = self.model

        # Initialize the agent graph
        self.app = self._create_agent_graph()

    def _llm_generate(self, prompt: str, max_new_tokens=512, temperature=0.0, stream=False) -> str:
        """Generate response using the LLM"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                streamer=streamer,
            )
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            output = "".join(list(streamer))
            thread.join()
            return output.strip()
        else:
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.0
                )[0]

            input_length = inputs.input_ids.shape[1]
            response_ids = out_ids[input_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            return response.strip()

    def _create_agent_graph(self):
        """Create the agent graph with your RAG logic"""
        graph = StateGraph(RAGState)
        graph.add_node("agent_node", self.agent_node)
        graph.set_entry_point("agent_node")

        def decision_router(state):
            return "answer" if state.get("réponse") else "terminate"

        graph.add_conditional_edges("agent_node", decision_router, {"answer": END, "terminate": END})
        return graph.compile()

    def tool_reformulate(self, question: str, docs: list[dict]) -> str:
        """Reformulate the question based on documents"""
        context = "\n".join(f"- {d['text'][:500]}..." for d in docs[:3])
        prompt = (
            "Tu es un agent de la fonction publique du Sénégal. "
            "La question suivante n’a pas obtenu de réponse suffisante. "
            "Voici les extraits les plus pertinents trouvés :\n"
            f"{context}\n\n"
            "Reformule la question pour cibler plus précisément l'information manquante, sans jamais modifier le sens "
            "en t’inspirant du vocabulaire des extraits si pertinent.\n"
            f"Question originale : {question}\n"
            "Nouvelle question :"
        )
        return self._llm_generate(prompt, max_new_tokens=128, temperature=0.0)

    def tool_judge(self, question: str, docs: list[dict]) -> str:
        """Judge if documents are sufficient to answer"""
        if not docs:
            return "REFORMULER"

        context = "\n\n".join(f"- {d['text']}" for d in docs)
        prompt = f"""Sur une échelle de 1-10, à quel point ces extraits permettent-ils de répondre à la question ? Réponds avec juste le chiffre.

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

    async def _stream_answer(self, question: str, docs: list[dict]) -> AsyncGenerator[str, None]:
        """Stream the answer generation token by token"""
        context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
        prompt = (
            "Réponds à la question en 5 phrases maximum.\n"
            "Inclue 1-2 citations EXACTES (entre guillemets) tirées des extraits.\n"
            "À la fin, ajoute une ligne 'Sources: ...' listant fichier:page.\n\n"
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
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
            await asyncio.sleep(0)

        thread.join()

        # Ajouter les sources
        sources = [f"{os.path.basename(d['source'])}:p.{d['page']+1}" for d in docs]
        yield f"\n\n📚 Sources: " + ", ".join(sorted(set(sources)))

    async def generate_stream(self, question: str, max_tokens: int = 512, temperature: float = 0.1):
        """Generate response with streaming using the agent"""
        question_current = question

        for step in range(1, settings.MAX_STEPS + 1):
            print(f"📚 Recherche d'informations (tentative {step})...")
            docs = self.vectorizer.hybrid_search(question_current, top_k=settings.TOP_K)

            verdict = self.tool_judge(question_current, docs)
            if verdict == "PERTINENT":
                
                async for token in self._stream_answer(question_current, docs):
                    yield token
                return
            else:
                if step == settings.MAX_STEPS:
                    yield "\n❌ Aucun document suffisamment pertinent trouvé.\n"
                    return

                
                print("🔄 Reformulation de la question...")
                question_current = self.tool_reformulate(question_current, docs)
                print(f"💡 Nouvelle question : {question_current}")

    def generate_sync(self, question: str) -> str:
        """Synchronous generation using the agent"""
        initial_state = {
            "question": question,
            "documents": [],
            "reformulé": False,
            "reformulation_count": 0,
            "réponse": None
        }
        final_state = self.app.invoke(initial_state)
        return final_state.get("réponse", "Aucune réponse générée.")

    def agent_node(self, state):
        """Main agent node with your RAG logic"""
        question = state["question"]

        for step in range(1, settings.MAX_STEPS + 1):
            docs = self.vectorizer.hybrid_search(question, top_k=settings.TOP_K)
            verdict = self.tool_judge(question, docs)

            if verdict == "PERTINENT":
                answer = self._llm_generate(
                    f"Réponds brièvement à la question : {question}",
                    max_new_tokens=256,
                    temperature=0.1,
                    stream=False
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
                        "réponse": "Je n’ai trouvé aucun document suffisamment pertinent."
                    }
                question = self.tool_reformulate(question, docs)

        return {
            "decision": "terminate",
            "documents": [],
            "réponse": "Arrêt sans réponse."
        }
