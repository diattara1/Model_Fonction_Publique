# tools/reformulator.py
class Reformulator:
    def __init__(self, generator):
        self.generator = generator

    def reformulate(self, question: str, docs: list[dict]) -> str:
        context = "\n".join(f"- {d['text'][:300]}..." for d in docs[:3])
        prompt = f"""
Tu es un assistant spécialisé dans la Fonction publique sénégalaise. 
La question posée n’a pas trouvé de réponse claire. 
Reformule-la en restant dans le cadre juridique et administratif du Sénégal.

Question originale : {question}
Extraits disponibles :
{context}

Nouvelle question :
"""

        return self.generator.simple_answer(prompt, max_new_tokens=128)
