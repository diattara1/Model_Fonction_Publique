# agents/rag_agent.py
from tools.retriever import Retriever
from tools.Judge import Judge
from tools.reformulator import Reformulator
from tools.answer_generator import AnswerGenerator
from models.generator import Generator
from models.vectorizer import Vectorizer


class RAGAgent:
    def __init__(self, vectorizer: Vectorizer, generator: Generator):
        self.retriever = Retriever(vectorizer)
        self.judge = Judge(generator)
        self.reformulator = Reformulator(generator)
        self.answer_generator = AnswerGenerator(generator)
        self.generator = generator

    async def answer(self, question: str, stream: bool = True):
        """
        Pipeline RAG LLM-based:
        - Si pas besoin de recherche → réponse directe (multi-tour)
        - Sinon → recherche + jugement + éventuelles reformulations
        - Fallback: proposer reformulation utilisateur + réponse provisoire
        """
        # 👉 notifier le front que le modèle réfléchit
        yield "🤔 Réflexion en cours…"

        need_retrieval = self._ask_need_retrieval(question)

        # === Cas 1 : Pas besoin de recherche ===
        if not need_retrieval:
            if stream:
                # ✅ utilise la mémoire multi-tour
                async for token in self.generator.stream_generate_mt(question):
                    yield token
            else:
                yield self.generator.simple_answer_mt(question)
            return

        # === Cas 2 : Recherche documentaire ===
        yield "📚 Recherche en cours..."
        current_q = question
        last_docs = []

        for step in range(3):  # max 3 reformulations
            docs = self.retriever.search(current_q)
            last_docs = docs
            verdict = self.judge.evaluate(current_q, docs)
            print("reformulation n:", step, " : ", current_q, " doc récupérés :", docs)

            if verdict == "PERTINENT":
                # ✅ réponse finale RAG avec mémoire
                async for token in self.answer_generator.stream_answer(
                    current_q, docs, use_history=True
                ):
                    yield token
                return
            else:
                current_q = self.reformulator.reformulate(current_q, docs)

        # === Cas 3 : Rien de parfaitement pertinent ===
        if last_docs:
            fallback_prompt = f"""
La question posée est : "{question}".

Les extraits trouvés ne répondent pas exactement, mais peuvent être liés.
Donne une réponse prudente et partielle basée uniquement sur ces extraits, 
sans inventer ni sortir du domaine de la fonction publique.
Termine par : "⚠️ Je n’ai pas trouvé de réponse parfaitement adaptée. Pouvez-vous reformuler votre question de façon plus précise ?"

Extraits:
{chr(10).join(f"- {d['text'][:400]}..." for d in last_docs[:3])}

Réponse provisoire:
"""
            if stream:
                async for token in self.generator.stream_generate_mt(fallback_prompt):
                    yield token
            else:
                yield self.generator.simple_answer_mt(fallback_prompt)
            return

        # === Cas 4 : Rien trouvé du tout ===
        yield "⚠️ Je n’ai rien trouvé de pertinent. Pouvez-vous préciser votre question ?"

    def _ask_need_retrieval(self, question: str) -> bool:
        """
        Demande au LLM si une recherche documentaire est nécessaire
        pour répondre correctement.
        """
        prompt = f"""
Tu es un assistant **spécialisé dans la Fonction publique sénégalaise**. 
Ton rôle est d’aider uniquement sur les questions juridiques, administratives et réglementaires liées à la fonction publique du Sénégal.

La question suivante nécessite-t-elle de consulter des documents juridiques/administratifs 
(pdfs, décrets, lois, statuts sénégalais) pour répondre correctement ?

Question: "{question}"

Réponds UNIQUEMENT par "OUI" ou "NON".
"""
        response = self.generator.simple_answer(prompt, max_new_tokens=3)
        return response.strip().upper().startswith("OUI")
