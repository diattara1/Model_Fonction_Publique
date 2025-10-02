# agents/rag_agent.py
import json
from tools.retriever import Retriever
from tools.judge import Judge
from tools.reformulator import Reformulator
from tools.answer_generator import AnswerGenerator
from models.generator import Generator
from models.generator import STATUS_RETRIEVAL_START, STATUS_DONE
from models.vectorizer import Vectorizer


class RAGAgent:
    """
    Agent RAG générique.

    Pipeline :
      1) Décider si une recherche documentaire est nécessaire (OUI/NON)
         → via simple_answer (sans mémoire) : rien n’est enregistré
      2) Si NON → génération directe (mémoire DB côté Generator)
      3) Si OUI → RAG :
         - émettre [[STATUS:RETRIEVAL_START]]
         - récupérer des extraits
         - juger la pertinence
         - éventuelle reformulation (≤ 3)
         - génération finale avec citations
      4) Émettre [[STATUS:DONE]] une seule fois à la fin
    """

    def __init__(self, vectorizer: Vectorizer, generator: Generator):
        self.retriever = Retriever(vectorizer)
        self.judge = Judge(generator)
        self.reformulator = Reformulator(generator)
        self.answer_generator = AnswerGenerator(generator)
        self.generator = generator

    # -----------------------------
    # Utilitaires
    # -----------------------------
    def _extract_question(self, raw_input: str) -> str:
        try:
            data = json.loads(raw_input)
            return (data.get("text") or "").strip()
        except (json.JSONDecodeError, AttributeError):
            return (raw_input or "").strip()

    def _ask_need_retrieval(self, question: str) -> bool:
        """
        IMPORTANT : on utilise simple_answer (pas *_mt) pour ne RIEN
        enregistrer en DB et ne pas polluer l’historique utilisateur.
        """
        prompt = f"""Tu es un assistant dans le domaine juridique/administratif.

Analyse la question suivante et réponds UNIQUEMENT par "OUI" ou "NON" :
- OUI si une recherche documentaire (textes officiels) est nécessaire
- NON sinon

Question: "{question}"

Réponse (OUI ou NON):"""
        resp = self.generator.simple_answer(prompt, max_new_tokens=5, temperature=0.0)
        return (resp or "").strip().upper().startswith("OUI")

    # -----------------------------
    # Main
    # -----------------------------
    async def answer(self, question: str, conv_id: int, stream: bool = True):
        clean_question = self._extract_question(question)
        if not clean_question:
            yield "⚠️ Question vide"
            yield STATUS_DONE
            return

        need_retrieval = self._ask_need_retrieval(clean_question)

        # === Cas 1 : sans recherche documentaire ===
        if not need_retrieval:
            if stream:
                async for token in self.generator.stream_generate_mt(clean_question, conv_id):
                    yield token
            else:
                yield self.generator.simple_answer_mt(clean_question, conv_id)
            yield STATUS_DONE
            return

        # === Cas 2 : avec recherche documentaire (RAG) ===
        yield STATUS_RETRIEVAL_START

        current_q = clean_question
        last_docs = []

        for _ in range(3):
            docs = self.retriever.search(current_q)
            last_docs = docs
            verdict = self.judge.evaluate(current_q, docs)

            if verdict == "PERTINENT":
                async for token in self.answer_generator.stream_answer(
                    current_q, docs, conv_id=conv_id, use_history=True
                ):
                    yield token
                yield STATUS_DONE
                return
            else:
                current_q = self.reformulator.reformulate(current_q, docs)

        # Rien de parfaitement pertinent mais des extraits existent
        if last_docs:
            fallback_prompt = f"""
La question posée est : "{clean_question}".

Les extraits trouvés ne répondent pas exactement, mais peuvent être liés.
Réponds prudemment en te basant uniquement sur ces extraits, sans inventer :

{chr(10).join(f"- {d['text'][:400]}..." for d in last_docs[:3])}
"""
            if stream:
                async for token in self.generator.stream_generate_mt(fallback_prompt, conv_id):
                    yield token
            else:
                yield self.generator.simple_answer_mt(fallback_prompt, conv_id)
            yield STATUS_DONE
            return

        # Aucun document pertinent
        yield "Je n'ai pas trouvé d'information pertinente dans mes documents. Pouvez-vous reformuler votre question ?"
        yield STATUS_DONE
