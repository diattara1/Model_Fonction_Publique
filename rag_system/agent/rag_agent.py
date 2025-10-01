from tools.retriever import Retriever
from tools.judge import Judge
from tools.reformulator import Reformulator
from tools.answer_generator import AnswerGenerator
from models.generator import Generator

class RAGAgent:
    def __init__(self, vectorizer, generator: Generator):
        self.retriever = Retriever(vectorizer)
        self.judge = Judge(generator)
        self.reformulator = Reformulator(generator)
        self.answer_generator = AnswerGenerator(generator)
        self.generator = generator

    async def answer(self, question: str, stream: bool = True):
        """Pipeline RAG : décision LLM-based"""
        # 1. Demander au LLM si une recherche est nécessaire
        need_retrieval = self._ask_need_retrieval(question)

        if not need_retrieval:
            # 🔹 Réponse directe sans retrieval
            if stream:
                async for token in self.generator.stream_simple(question):
                    yield token
            else:
                yield self.generator.simple_answer(question)
            return

        # 2. Recherche + RAG
        current_q = question
        for step in range(3):  # nombre max de reformulations
            docs = self.retriever.search(current_q)
            verdict = self.judge.evaluate(current_q, docs)

            if verdict == "PERTINENT":
                async for token in self.answer_generator.stream_answer(current_q, docs):
                    yield token
                return
            else:
                current_q = self.reformulator.reformulate(current_q, docs)

        yield "❌ Impossible de trouver une réponse pertinente."

    def _ask_need_retrieval(self, question: str) -> bool:
        """Demande au LLM si une recherche documentaire est nécessaire"""
        prompt = f"""
Tu es un assistant. La question suivante nécessite-t-elle
de consulter des documents juridiques/administratifs
(pdfs, décrets, lois, statuts) pour répondre correctement ?

Question: "{question}"

Réponds UNIQUEMENT par "OUI" ou "NON".
"""
        response = self.generator.simple_answer(prompt, max_new_tokens=3)
        return response.strip().upper().startswith("OUI")
