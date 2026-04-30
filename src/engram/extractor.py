from src.llm.cloud import cloud_chat
from src.engram.schemas import TripleExtraction
import json
import json
import re


SYSTEM_PROMPT = """
Extract knowledge graph triples from text.

Return ONLY valid JSON:
{"triples":[{"subject":{"label":"","name":"","properties":{}},"relation":"","object":{"label":"","name":"","properties":{}},"relation_properties":{}}]}

Rules:
- Use only explicit info (no guessing)
- No extra text
- No duplicates
- Consistent naming

Labels:
Person, Company, City, Country, Skill, Role, Organization, Technology

Relations:
WORKS_AT, LIVES_IN, HAS_SKILL, LOVES, BUILDS, IS_A
"""

def extract_triples(text: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> TripleExtraction:
    content = cloud_chat(
        messages=[{"role": "user", "content": text}],
        model=model,
        system_prompt=SYSTEM_PROMPT,
    )

    data = parse_llm_json(content=content)
    data["triples"] = normalize_triples(data["triples"])
    return TripleExtraction(**data)

def normalize_triples(triples):
    seen = set()
    unique = []

    for t in triples:
        key = (
            t["subject"]["name"].lower(),
            t["relation"],
            t["object"]["name"].lower()
        )
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique

def parse_llm_json(content: str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # extract first valid JSON object
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No valid JSON found in LLM output")
if __name__ == "__main__":
    text = """
    Kartikey is a backend developer at EXL.
    He lives in Kanpur and loves building AI agents.
    """

    extraction = extract_triples(text)

    print("Extracted:")
    print(extraction.model_dump_json(indent=2))