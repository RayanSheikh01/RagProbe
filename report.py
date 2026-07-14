    

from models import Report


def build_report(scores, versions) -> Report:
    return Report(scores, versions)


def render_markdown(report: Report) -> str:
    md = "# Evaluation Report\n\n"
    md += "## Versions\n"
    for component, version in report.versions.items():
        md += f"- **{component}**: {version}\n"
    
    md += "\n## Scores\n"
    for score in report.scores:
        md += f"- **Probe ID**: {score.probe_id}\n"
        md += f"  - **Scorer**: {score.scorer}\n"
        md += f"  - **Value**: {score.value:.4f}\n"
        md += f"  - **Passed**: {'Yes' if score.passed else 'No'}\n"
        md += f"  - **Detail**: {score.detail}\n\n"
    
    return md

