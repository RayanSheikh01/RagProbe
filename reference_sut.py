class ReferenceSUT:
    """One working SUT so you have something to run against.
    Any object with .index()/.query() works in the runner — no base class."""
    
    def __init__(self, embed_fn, llm_fn, top_k: int = 5):
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.top_k = top_k
        self.docs = []
        self.embeddings = []
    
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
    
    def cosine_similarity(self, vec_a, vec_b) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        return dot_product / (norm_a * norm_b + 1e-10)  # Add small value to avoid division by zero
    
    