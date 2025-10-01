import os

class AnswerGenerator:
    def __init__(self, generator):
        self.generator = generator

    async def stream_answer(self, question: str, docs: list[dict]):
        context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
        prompt = f"""
Réponds en tant qu’assistant spécialisé Fonction publique du Sénégal.
Appuie-toi EXCLUSIVEMENT sur les extraits ci-dessous. Max 5 phrases.
Cite le texte pertinent entre guillemets. Si rien de précis, dis-le et propose une reformulation.

Question: {question}

Extraits:
{context}

Réponse:
"""


        async for token in self.generator.stream_generate(prompt):
            yield token

        # Ajout des sources
        sources = [f"{os.path.basename(d['source'])}:p.{d['page']+1}" for d in docs]
        yield f"\n\n📚 Sources: " + ", ".join(sorted(set(sources)))
