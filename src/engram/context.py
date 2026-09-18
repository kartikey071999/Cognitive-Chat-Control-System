import re
from src.engram.graph import KnowledgeGraph


class KGContext:
    def __init__(self):
        self.context = ""

    def update(self, graph: KnowledgeGraph, text: str):
        keywords = self._extract_keywords(text)
        matches = []

        for keyword in keywords:
            for node, data in graph.get_nodes():
                if keyword.lower() in node.lower():
                    edges = [
                        e for e in graph.get_edges() if e[0] == node or e[1] == node
                    ]
                    relations = [
                        f"{e[0]} -{e[2].get('relation', '?')}-> {e[1]}" for e in edges
                    ]
                    matches.append(
                        f"{node} ({data.get('label', '?')}): {', '.join(relations)}"
                    )

        self.context = "\n".join(matches) if matches else ""
        return self.context

    def _extract_keywords(self, text: str):
        stop_words = {
            "i",
            "me",
            "my",
            "you",
            "your",
            "we",
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "do",
            "does",
            "did",
            "have",
            "has",
            "had",
            "will",
            "would",
            "can",
            "could",
            "should",
            "what",
            "who",
            "where",
            "when",
            "how",
            "which",
            "that",
            "this",
            "it",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "about",
            "not",
            "no",
            "but",
            "or",
            "and",
            "if",
            "so",
            "just",
            "also",
            "than",
            "then",
            "very",
            "too",
            "much",
            "more",
        }
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text)
        return [w for w in words if w.lower() not in stop_words]

    def get(self):
        return self.context

    def clear(self):
        self.context = ""
