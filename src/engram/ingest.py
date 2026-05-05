from src.db.neo4j_client import Neo4jClient
from src.engram.schemas import TripleExtraction


class GraphIngestor:
    def __init__(self):
        self.db = Neo4jClient()

    def ingest(self, extraction: TripleExtraction):
        for triple in extraction.triples:
            # merge subject
            self.db.merge_node(
                triple.subject.label, triple.subject.name, triple.subject.properties
            )

            # merge object
            self.db.merge_node(
                triple.object.label, triple.object.name, triple.object.properties
            )

            # merge relationship
            self.db.merge_relationship(
                triple.subject.label,
                triple.subject.name,
                triple.relation,
                triple.object.label,
                triple.object.name,
                triple.relation_properties,
            )

    def close(self):
        self.db.close()
