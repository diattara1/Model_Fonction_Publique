import os

class AnswerGenerator:
    def __init__(self, generator):
        self.generator = generator

    async def stream_answer(self, question: str, docs: list[dict]):
        context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
        prompt = (
            "Réponds à la question en 5 phrases maximum.\n"
            "Inclue 1-2 citations EXACTES (entre guillemets) tirées des extraits.\n"
            "Ajoute une ligne 'Sources: ...' listant fichier:page.\n\n"
            f"Question:\n{question}\n\nExtraits:\n{context}\n\nRéponse:"
        )

        async for token in self.generator.stream_generate(prompt):
            yield token

        # Ajout des sources
        sources = [f"{os.path.basename(d['source'])}:p.{d['page']+1}" for d in docs]
        yield f"\n\n📚 Sources: " + ", ".join(sorted(set(sources)))
