from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging
from config.config import settings

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    def _run_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            try:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
            except Exception as e:
                logger.error(f"Error executing query: {str(e)}")
                raise
    
    def get_entity_by_name(self, entity_name: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (e)
        WHERE e.name CONTAINS $entity_name
        RETURN e
        """
        return self._run_query(query, {"entity_name": entity_name})
    
    def get_relationships(self, entity_id: str, max_hops: int = None) -> List[Dict[str, Any]]:
        max_hops = max_hops or settings.MAX_HOPS
        query = f"""
        MATCH path = (e)-[*1..{max_hops}]-(related)
        WHERE e.id = $entity_id
        RETURN path
        """
        return self._run_query(query, {"entity_id": entity_id})
    
    def create_entity(self, entity_type: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        query = f"""
        CREATE (e:{entity_type} $properties)
        RETURN e
        """
        return self._run_query(query, {"properties": properties})[0]
    
    def create_relationship(self, from_id: str, to_id: str, relationship_type: str, properties: Dict[str, Any] = None) -> Dict[str, Any]:
        query = f"""
        MATCH (from), (to)
        WHERE from.id = $from_id AND to.id = $to_id
        CREATE (from)-[r:{relationship_type} $properties]->(to)
        RETURN r
        """
        return self._run_query(query, {
            "from_id": from_id,
            "to_id": to_id,
            "properties": properties or {}
        })[0] 