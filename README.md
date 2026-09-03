# RAGProbe

Injects counterfactual facts into a document corpus, asks a RAG system about
them, and scores whether it retrieved the right chunk and answered from it.

Counterfactual facts (invented entities/numbers) mean a correct answer can only
come from retrieval, not from the model's pretraining.

## Install

```bash
pip install numpy pyyaml
```

## Usage

```bash
python cli.py probes.yaml --docs ./corpus --top-k 5
```

- `probes.yaml` — list of probes (see below)
- `--docs` — directory of `.txt` files used as the base corpus
- `--top-k` — contexts retrieved per question (default 5)
- `--no-judge` — skip LLM-judged groundedness

Prints a markdown report to stdout.

### probes.yaml

```yaml
- id: p1
  fact: "The Zylark reactor produces 4.7 gigawatts."
  question: "How much power does the Zylark reactor produce?"
  expected_answer: "4.7 gigawatts"
```

## Scorers

| Scorer | Measures |
|---|---|
| `retrieval` | reciprocal rank of the injected gold chunk within top-k |
| `correctness` | expected answer present in the SUT's answer |
| `groundedness` | LLM judge: is every claim in the answer entailed by the retrieved contexts |

The judge must be a different model than the system under test — no self-grading.
Judge calls are cached and run at temperature 0.

## Files

`models.py` dataclasses · `injector.py` fact injection · `reference_sut.py`
in-memory embed+retrieve SUT · `runner.py` orchestration · `scorers.py` ·
`judge.py` entailment judge · `report.py` markdown · `cli.py` entrypoint.

`IMPLEMENTATION_PLAN.md` has the design rationale.
