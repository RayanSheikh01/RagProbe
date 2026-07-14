class ReferenceSUT:
    """One working SUT so you have something to run against.
    Any object with .index()/.query() works in the runner — no base class."""
    
    def __init__(self, embed_fn, llm_fn, top_k: int = 5) ...
    
    def index(self, docs: list[str]) -> None:
        """Index the docs in memory."""
        self.docs = docs
        self.embeddings = [self.embed_fn(doc) for doc in docs]
        
    def query(self, question: str) -> list[tuple[str, float]]:
        """Return top-k docs with scores."""
        q_emb = self.embed_fn(question)
        scores = [(doc, self.cosine_similarity(q_emb, emb)) for doc, emb in zip(self.docs, self.embeddings)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:self.top_k]
    
    