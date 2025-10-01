class Judge:
    def __init__(self, generator):
        self.generator = generator

    def evaluate(self, question: str, docs: list[dict]) -> str:
        if not docs:
            return "REFORMULER"

        context = "\n\n".join(f"- {d['text']}" for d in docs)
        prompt = f"""Sur une échelle de 1-10, à quel point ces extraits permettent-ils
de répondre à la question ? Réponds uniquement par un chiffre.

Question: {question}

Extraits:
{context}

Score (1-10):"""

        score_text = self.generator.simple_answer(prompt, max_new_tokens=3)
        try:
            score = int(score_text.strip())
            return "PERTINENT" if score >= 6 else "REFORMULER"
        except:
            return "REFORMULER"
