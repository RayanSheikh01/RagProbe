
from judge import judge_entailment
from models import Probe, RunResult, Score

def score_retrieval(rr: RunResult, k: int = 5) -> Score:
    
    gold = rr.gold_context
    retrieved = rr.retrieval.retrieved_contexts[:k]
    
    rank = None
    for idx, context in enumerate(retrieved, start=1):
        if gold in context:
            rank = idx
            break
    
    value = 1.0 / rank if rank is not None else 0.0
    passed = rank is not None and rank <= k
    detail = {"rank": rank, "hit_at_k": passed}
    
    return Score(
        probe_id=rr.probe_id,
        scorer="retrieval",
        value=value,
        passed=passed,
        detail=detail
    )
    
def score_correctness(rr, probe: Probe) -> Score:
    answer = rr.retrieval.answer
    expected = probe.expected_answer
    
    passed = answer.strip().lower() == expected.strip().lower()
    value = 1.0 if passed else 0.0
    detail = {"expected": expected, "actual": answer}
    
    return Score(
        probe_id=rr.probe_id,
        scorer="correctness",
        value=value,
        passed=passed,
        detail=detail
    
    )
    
    
def score_groundedness(rr, judge__llm_fn) -> Score:
    answer = rr.retrieval.answer
    contexts = rr.retrieval.retrieved_contexts
    
    result = judge_entailment(answer, contexts, judge__llm_fn)
    
    value = 1.0 if result["grounded"] else 0.0
    passed = result["grounded"]
    detail = {
        "rationale": result.get("rationale", ""),
        "supporting_idx": result.get("supporting_idx", [])
    }
    
    return Score(
        probe_id=rr.probe_id,
        scorer="groundedness",
        value=value,
        passed=passed,
        detail=detail
    )