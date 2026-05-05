from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
with driver.session() as session:
    session.run("""
        MERGE (p:Person {name: "Kartikey"})
        SET p.role = "Backend Developer"
    """)
with driver.session() as session:
    result = session.run("RETURN 'connected' AS msg")
    print(result.single()["msg"])

driver.close()
