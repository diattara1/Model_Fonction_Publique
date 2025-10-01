class Reformulator:
    def __init__(self, generator):
        self.generator = generator

    def reformulate(self, question: str, docs: list[dict]) -> str:
        context = "\n".join(f"- {d['text'][:300]}..." for d in docs[:3])
        prompt = (
            "Tu es un agent de la fonction publique du Sénégal.\n"
            "La question suivante n’a pas obtenu de réponse suffisante.\n"
            f"Voici les extraits :\n{context}\n\n"
            "Reformule la question de manière plus ciblée, sans en changer le sens.\n"
            f"Question originale : {question}\n"
            "Nouvelle question :"
        )
        return self.generator.simple_answer(prompt, max_new_tokens=128)
