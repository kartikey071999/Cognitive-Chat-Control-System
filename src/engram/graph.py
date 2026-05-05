import networkx as nx
from src.engram.extractor import extract_triples
from src.engram.schemas import TripleExtraction


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_extraction(self, extraction: TripleExtraction):
        for triple in extraction.triples:
            sub = triple.subject
            obj = triple.object

            self.graph.add_node(sub.name, label=sub.label, **(sub.properties or {}))
            self.graph.add_node(obj.name, label=obj.label, **(obj.properties or {}))
            self.graph.add_edge(
                sub.name,
                obj.name,
                relation=triple.relation,
                **(triple.relation_properties or {}),
            )

    def ingest_chat(self, text: str):
        try:
            extraction = extract_triples(text)
            self.add_extraction(extraction)
            return extraction
        except Exception as e:
            print(f"[KG] extraction failed: {e}")
            return None

    def get_nodes(self):
        return list(self.graph.nodes(data=True))

    def get_edges(self):
        return list(self.graph.edges(data=True))

    def get_neighbors(self, name: str):
        if name in self.graph:
            return list(self.graph.successors(name)) + list(
                self.graph.predecessors(name)
            )
        return []

    def __repr__(self):
        return f"KnowledgeGraph(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"

    def exit(self):
        self.graph.clear()
        del self.graph
        self.graph = None
