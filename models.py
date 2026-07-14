from dataclasses import dataclass, field

@dataclass
class Probe:
    id: str
    fact: str              # counterfactual statement (invented entity/number) — Risk 3
    question: str
    expected_answer: str
    # distractors deferred to phase 2 — see Rung 7

@dataclass
class RetrievalResult:
    question: str
    retrieved_contexts: list[str]   # rank-ordered, rank matters
    answer: str
    raw: dict = field(default_factory=dict)

@dataclass
class RunResult:
    probe_id: str
    gold_context: str               # text of the injected chunk
    retrieval: RetrievalResult

@dataclass
class Score:
    probe_id: str
    scorer: str
    value: float                    # 0..1
    passed: bool
    detail: dict = field(default_factory=dict)

@dataclass
class Report:
    scores: list[Score]
    versions: dict                  # {sut_model, judge_model, embed_model} — Risk 4
    # aggregates/failures NOT stored — derived in report.render_markdown from scores