from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class Node(BaseModel):
    label: str
    name: str
    properties: Optional[Dict] = Field(default_factory=dict)


class Triple(BaseModel):
    subject: Node
    relation: str
    object: Node
    relation_properties: Optional[Dict] = Field(default_factory=dict)


class TripleExtraction(BaseModel):
    triples: List[Triple]