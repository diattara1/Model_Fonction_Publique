class Retriever:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def search(self, query: str, top_k: int = 5):
        return self.vectorizer.hybrid_search(query, top_k=top_k)
