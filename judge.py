import hashlib
import json

_CACHE: dict[str, dict] = {}

_PROMPT = """You are a strict entailment judge. Given an ANSWER and numbered CONTEXTS, decide if every claim in the ANSWER is fully supported by the CONTEXTS.

Return ONLY a JSON object:
{{"grounded": <true|false>, "rationale": "<one sentence>", "supporting_idx": [<indices of contexts that support the answer>]}}

ANSWER:
{answer}

CONTEXTS:
{contexts}"""


def _parse(text: str) -> dict:
    """Pull the JSON object out of a model response; raises on failure."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    obj = json.loads(text[start:end + 1])
    return {
        "grounded": bool(obj["grounded"]),
        "rationale": str(obj.get("rationale", "")),
        "supporting_idx": [int(i) for i in obj.get("supporting_idx", [])],
    }


def judge_entailment(answer: str, contexts: list[str], judge__llm_fn) -> dict:
    """Is `answer` fully supported by `contexts`?
    Return {grounded: bool, rationale: str, supporting_idx: list[int]}.
    - use a DIFFERENT model than the SUT (Risk 1, no self-grading).
    - cache on hash(answer + contexts + model) (Risk 4/5).
    - temperature 0; retry+parse-guard the JSON."""
    # model id: judge__llm_fn is opaque, so key on its identity. Caller must pin temp=0.
    model_id = getattr(judge__llm_fn, "__qualname__", repr(judge__llm_fn))
    key = hashlib.sha256(
        json.dumps([answer, contexts, model_id]).encode()
    ).hexdigest()
    if key in _CACHE:
        return _CACHE[key]

    numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    prompt = _PROMPT.format(answer=answer, contexts=numbered)

    last_err = None
    for _ in range(2):  # retry once on unparseable output
        try:
            result = _parse(judge__llm_fn(prompt))
            _CACHE[key] = result
            return result
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_err = e
    # both attempts failed: fail closed (not grounded) rather than crash the run
    result = {"grounded": False, "rationale": f"judge parse failed: {last_err}", "supporting_idx": []}
    _CACHE[key] = result
    return result


if __name__ == "__main__":
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return 'sure! {"grounded": true, "rationale": "ctx 1 states it", "supporting_idx": [1]}'

    r = judge_entailment("Paris is the capital", ["Berlin is big", "Paris is the capital of France"], fake_llm)
    assert r == {"grounded": True, "rationale": "ctx 1 states it", "supporting_idx": [1]}, r

    # cache hit: no second llm call
    judge_entailment("Paris is the capital", ["Berlin is big", "Paris is the capital of France"], fake_llm)
    assert len(calls) == 1, calls

    # unparseable -> fail closed after retry
    r2 = judge_entailment("x", ["y"], lambda p: "no json here")
    assert r2["grounded"] is False and r2["supporting_idx"] == [], r2

    print("ok")
