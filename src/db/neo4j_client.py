from neo4j import GraphDatabase
import networkx as nx
from typing import List, Dict, Any
import torch

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def find_similar_nodes(self, embedding: torch.Tensor, threshold: float) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n) RETURN n LIMIT 10"
            )
            return [record["n"] for record in result]
            
    def build_subgraph(self, nodes: List[Dict[str, Any]], max_hops: int) -> nx.Graph:
        graph = nx.Graph()
        
        with self.driver.session() as session:
            for node in nodes:
                graph.add_node(node.id, **node)
                
            for node in nodes:
                result = session.run(
                    """
                    MATCH (n)-[r]-(m)
                    WHERE id(n) = $node_id
                    RETURN r, m
                    LIMIT 10
                    """,
                    node_id=node.id
                )
                
                for record in result:
                    rel = record["r"]
                    target = record["m"]
                    graph.add_edge(node.id, target.id, **rel)
                    
        return graph
        
    def add_node(self, properties: Dict[str, Any]) -> str:
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (n:Entity $properties)
                RETURN id(n)
                """,
                properties=properties
            )
            return result.single()[0]
            
    def add_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any] = None) -> None:
        if properties is None:
            properties = {}
            
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (a), (b)
                WHERE id(a) = $source_id AND id(b) = $target_id
                CREATE (a)-[r:{rel_type} $properties]->(b)
                """,
                source_id=source_id,
                target_id=target_id,
                properties=properties
            ) 