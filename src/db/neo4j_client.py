from neo4j import GraphDatabase, exceptions
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


class Neo4jClient:
    def __init__(self, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        self.database = database

    def close(self):
        if self.driver:
            self.driver.close()

    # ---------------------------
    # Internal transaction methods
    # ---------------------------

    @staticmethod
    def _read_tx(tx, query, params):
        result = tx.run(query, params or {})
        return [record.data() for record in result]

    @staticmethod
    def _write_tx(tx, query, params):
        tx.run(query, params or {})

    # ---------------------------
    # Public API
    # ---------------------------

    def run_query(self, query: str, params: dict = None):
        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_read(self._read_tx, query, params)
        except exceptions.Neo4jError as e:
            print(f"[Neo4j READ ERROR] {e}")
            return []

    def execute_write(self, query: str, params: dict = None):
        try:
            with self.driver.session(database=self.database) as session:
                session.execute_write(self._write_tx, query, params)
        except exceptions.Neo4jError as e:
            print(f"[Neo4j WRITE ERROR] {e}")

    def test_connection(self):
        result = self.run_query("RETURN 'connected' AS msg")
        return result[0]["msg"] if result else "failed"

    # ---------------------------
    # Graph Helpers (IMPORTANT)
    # ---------------------------

    def merge_node(self, label: str, name: str, properties: dict = None):
        label = label or "Entity"
        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET n += $props
        """
        self.execute_write(query, {
            "name": name,
            "props": properties or {}
        })

    def merge_relationship(
        self,
        from_label: str,
        from_name: str,
        rel_type: str,
        to_label: str,
        to_name: str,
        rel_props: dict = None
    ):
        from_label = from_label or "Entity"
        to_label = to_label or "Entity"
        query = f"""
        MATCH (a:{from_label} {{name: $from_name}})
        MATCH (b:{to_label} {{name: $to_name}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        """
        self.execute_write(query, {
            "from_name": from_name,
            "to_name": to_name,
            "props": rel_props or {}
        })


if __name__ == "__main__":
    db = Neo4jClient()

    print(db.test_connection())

    db.merge_node("Person", "Kartikey", {
        "role": "Backend Developer",
        "interest": "AI Agents"
    })

    db.merge_node("Company", "EXL")

    db.merge_relationship(
        "Person", "Kartikey",
        "WORKS_AT",
        "Company", "EXL"
    )

    print(db.run_query("MATCH (n) RETURN n LIMIT 5"))

    db.close()