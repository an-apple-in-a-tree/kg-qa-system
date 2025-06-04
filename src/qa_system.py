from typing import Dict, Any, List
import logging
from database.neo4j_client import Neo4jClient
from nlp.processor import NLPProcessor
from config.config import settings

logger = logging.getLogger(__name__)

class QASystem:
    def __init__(self):
        self.db_client = Neo4jClient()
        self.nlp_processor = NLPProcessor()
    
    def process_question(self, question: str) -> Dict[str, Any]:
        try:
            # Process the question using NLP
            processed_question = self.nlp_processor.process_question(question)
            
            # Extract entities from the question
            entities = processed_question["entities"]
            if not entities:
                return {
                    "answer": "I couldn't identify any specific entities in your question. Could you please rephrase it?",
                    "confidence": 0.0
                }
            
            # Search for entities in the knowledge graph
            relevant_entities = []
            for entity in entities:
                db_entities = self.db_client.get_entity_by_name(entity["text"])
                relevant_entities.extend(db_entities)
            
            if not relevant_entities:
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base for your question.",
                    "confidence": 0.0
                }
            
            # Get relationships for the most relevant entity
            primary_entity = relevant_entities[0]
            relationships = self.db_client.get_relationships(primary_entity["id"])
            
            # Generate answer based on question type and available information
            answer = self._generate_answer(
                question,
                processed_question["question_type"],
                primary_entity,
                relationships
            )
            
            return {
                "answer": answer,
                "confidence": self._calculate_confidence(processed_question, relevant_entities),
                "source_entities": relevant_entities
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "confidence": 0.0
            }
    
    def _generate_answer(
        self,
        question: str,
        question_type: str,
        primary_entity: Dict[str, Any],
        relationships: List[Dict[str, Any]]
    ) -> str:
        if question_type == "factual":
            return self._generate_factual_answer(primary_entity, relationships)
        elif question_type == "explanatory":
            return self._generate_explanatory_answer(primary_entity, relationships)
        elif question_type == "temporal_spatial":
            return self._generate_temporal_spatial_answer(primary_entity, relationships)
        else:
            return self._generate_general_answer(primary_entity, relationships)
    
    def _generate_factual_answer(self, entity: Dict[str, Any], relationships: List[Dict[str, Any]]) -> str:
        # Implementation for factual questions
        return f"Based on the information in our knowledge base, {entity.get('name', 'this entity')} is associated with {len(relationships)} related entities."
    
    def _generate_explanatory_answer(self, entity: Dict[str, Any], relationships: List[Dict[str, Any]]) -> str:
        # Implementation for explanatory questions
        return f"Here's what we know about {entity.get('name', 'this entity')}: {self._format_relationships(relationships)}"
    
    def _generate_temporal_spatial_answer(self, entity: Dict[str, Any], relationships: List[Dict[str, Any]]) -> str:
        # Implementation for temporal/spatial questions
        return f"Regarding the time and location of {entity.get('name', 'this entity')}: {self._format_relationships(relationships)}"
    
    def _generate_general_answer(self, entity: Dict[str, Any], relationships: List[Dict[str, Any]]) -> str:
        # Implementation for general questions
        return f"Here's what we know about {entity.get('name', 'this entity')}: {self._format_relationships(relationships)}"
    
    def _format_relationships(self, relationships: List[Dict[str, Any]]) -> str:
        if not relationships:
            return "No specific relationships found."
        return " ".join([f"{rel.get('type', 'related to')} {rel.get('target', 'another entity')}" for rel in relationships[:3]])
    
    def _calculate_confidence(self, processed_question: Dict[str, Any], relevant_entities: List[Dict[str, Any]]) -> float:
        # Simple confidence calculation based on entity match and question processing
        entity_match_score = len(relevant_entities) / max(1, len(processed_question["entities"]))
        return min(1.0, entity_match_score * 0.8)  # Cap at 0.8 to account for uncertainty
    
    def close(self):
        self.db_client.close() 