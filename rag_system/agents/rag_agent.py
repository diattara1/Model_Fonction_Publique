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
        Pipeline RAG LLM-based.
        - Notifie "Réflexion en cours" (enable_thinking).
        - Si pas besoin de recherche → réponse directe.
        - Sinon → recherche + jugement + (jusqu'à 3) reformulations.
        - Si rien de pertinent → propose reformulation + réponse provisoire, sans hors-sujet.
        """
        # === 0) notifier réflexion (thinking) ===
        yield "🤔 Réflexion en cours...\n"

        # === 1) décision : faut-il rechercher ? ===
        need_retrieval = self._ask_need_retrieval(question)

        # === Cas A : Pas besoin de recherche → réponse directe ===
        if not need_retrieval:
            if stream:
                async for token in self.generator.stream_generate(question):
                    yield token
            else:
                yield self.generator.simple_answer(question)
            return

        # === Cas B : Recherche documentaire ===
        yield "📚 Recherche en cours...\n"

        current_q = question
        last_docs = []  # garder les derniers documents récupérés (fallback)
        for step in range(3):  # max 3 reformulations
            docs = self.retriever.search(current_q)
            last_docs = docs

            verdict = self.judge.evaluate(current_q, docs)
            # Logs serveurs utiles
            print(f"[RAG] Étape {step+1} — question: {current_q}")
            print(f"[RAG] Docs récupérés: {len(docs)} | Verdict: {verdict}")

            if verdict == "PERTINENT":
                # Génération finale (streamée)
                if stream:
                    async for token in self.answer_generator.stream_answer(current_q, docs):
                        yield token
                else:
                    yield self.answer_generator.simple_answer(current_q, docs)
                return
            else:
                # Reformuler et réessayer
                current_q = self.reformulator.reformulate(current_q, docs)

        # === Cas C : Rien de parfaitement pertinent après 3 tentatives ===
        # Politique : ne jamais faire de hors-sujet. On propose reformulation + réponse prudente.
        if last_docs:
            # 1) informer l’utilisateur
            yield (
                "\n⚠️ Je n’ai pas trouvé de disposition correspondant exactement à votre demande. "
                "Pouvez-vous reformuler votre question de façon plus précise (ex. type de société, rôle : associé/gérant, "
                "contexte : cumul d’activités, conflit d’intérêts, etc.) ?\n"
                "↳ En attendant, voici des éléments susceptibles d’éclairer le sujet :\n\n"
            )

            # 2) réponse provisoire strictement basée sur les extraits les plus proches
            fallback_prompt = (
                "Tu es un assistant spécialisé dans la Fonction publique du Sénégal. "
                "Les extraits ci-dessous peuvent être liés au sujet mais ne répondent pas exactement à la question. "
                "Fais une réponse prudente et partielle STRICTEMENT basée sur ces extraits, sans inventer, "
                "et sans sortir du domaine de la fonction publique. "
                "Si l’information manque, dis-le clairement et suggère une reformulation précise.\n\n"
                f"Question initiale : {question}\n\n"
                "Extraits (ne cite que s’ils sont pertinents, sinon dis que l’info manque) :\n"
                + "\n".join(f"- {d['text'][:400]}..." for d in last_docs[:3])
                + "\n\nRéponse provisoire :"
            )

            if stream:
                async for token in self.generator.stream_generate(fallback_prompt):
                    yield token
            else:
                yield self.generator.simple_answer(fallback_prompt)
        else:
            # Aucun doc exploitable du tout
            yield (
                "⚠️ Je n’ai trouvé aucun extrait exploitable. "
                "Pouvez-vous reformuler votre question (ex. type de société, rôle envisagé, cadre exact) ?"
            )

    # === Décide si une recherche documentaire est nécessaire ===
    def _ask_need_retrieval(self, question: str) -> bool:
        prompt = f"""
Tu es un assistant **spécialisé dans la Fonction publique sénégalaise**. 
Tu réponds uniquement sur des questions juridiques, administratives et réglementaires du Sénégal.

La question suivante nécessite-t-elle de consulter des documents juridiques/administratifs 
(PDFs, décrets, lois, statuts sénégalais) pour répondre correctement ?

Question: "{question}"

Réponds UNIQUEMENT par "OUI" ou "NON".
"""
        response = self.generator.simple_answer(prompt, max_new_tokens=3)
        return response.strip().upper().startswith("OUI")
