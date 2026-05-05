from src.engram.extractor import extract_triples
from src.engram.ingest import GraphIngestor

if __name__ == "__main__":
    text = """
    Kartikey is a backend developer at EXL.
    He lives in Kanpur and loves building AI agents.
    """

    extraction = extract_triples(text)

    print("Extracted:")
    print(extraction.model_dump_json(indent=2))

    ingestor = GraphIngestor()
    ingestor.ingest(extraction)
    ingestor.close()
