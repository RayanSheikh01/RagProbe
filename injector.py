    

from models import Probe


def inject(base_docs: list[str], probes: list[Probe]) -> tuple[list[str], dict[str, str]]:
    augmented_docs = base_docs.copy()
    provenance = {}
    
    for probe in probes:
        # Create a self-contained document for each fact
        fact_doc = f"This is a self-contained document about {probe.fact}."
        augmented_docs.append(fact_doc)
        provenance[probe.id] = fact_doc

    return augmented_docs, provenance

