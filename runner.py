from models import Probe, RunResult, RetrievalResult

def run(probes, base_docs, sut) -> list[RunResult]:
    augmented, provenance = inject(base_docs, probes)
    sut.index(augmented)
    results = []
    
    for probe in probes:
        retrieved_contexts = sut.query(probe.question)
        answer = retrieved_contexts[0][0] if retrieved_contexts else ""
        rr = RunResult(
            probe_id=probe.id,
            gold_context=provenance[probe.id],
            retrieval=RetrievalResult(
                question=probe.question,
                retrieved_contexts=[doc for doc, _ in retrieved_contexts],
                answer=answer
            )
        )
        results.append(rr)
    
    return results