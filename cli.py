def main():
    import argparse
    import yaml
    from pathlib import Path
    from models import Probe, RunResult, Score, Report
    from reference_sut import ReferenceSUT
    from runner import run
    from scorers import score_retrieval, score_correctness, score_groundedness
    from report import build_report, render_markdown

    parser = argparse.ArgumentParser(description="Run probes against a SUT and generate a report.")
    parser.add_argument("probes_file", type=Path, help="Path to the probes YAML file.")
    parser.add_argument("--docs", type=Path, required=True, help="Path to the directory containing documents to index.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top documents to retrieve.")
    parser.add_argument("--no-judge", action="store_true", help="Skip groundedness scoring.")
    
    args = parser.parse_args()

    # Load probes from YAML file
    with open(args.probes_file, "r") as f:
        probes_data = yaml.safe_load(f)
    
    probes = [Probe(**probe) for probe in probes_data]

    # Load documents from the specified directory
    docs = [doc.read_text() for doc in args.docs.glob("*.txt")]

    # Initialize the SUT (System Under Test)
    sut = ReferenceSUT(embed_fn=lambda x: [1.0] * 768, llm_fn=lambda x: "dummy answer", top_k=args.top_k)
    
    # Run the probes against the SUT
    run_results = run(probes, docs, sut)

    # Score each run result
    scores = []
    for rr in run_results:
        scores.append(score_retrieval(rr, k=args.top_k))
        scores.append(score_correctness(rr, next(p for p in probes if p.id == rr.probe_id)))
        if not args.no_judge:
            scores.append(score_groundedness(rr, judge__llm_fn=lambda x: {"grounded": True}))

    # Build and render the report
    versions = {
        "ragprobe": "1.0.0",
        "reference_sut": "1.0.0"
    }
    
    report = build_report(scores, versions)
    markdown_report = render_markdown(report)
    
    print(markdown_report)

if __name__ == "__main__":
    main()