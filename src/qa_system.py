from typing import Tuple, List
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import networkx as nx
from config.config import settings
from src.db.neo4j_client import Neo4jClient
from src.nlp.processor import NLPProcessor

class QASystem:
    def __init__(self):
        self.nlp = spacy.load(settings.SPACY_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModel.from_pretrained(settings.MODEL_NAME)
        
        self.db = Neo4jClient(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD
        )
        
        self.nlp_processor = NLPProcessor(self.nlp, self.tokenizer, self.model)
    
    def answer_question(self, question: str, max_hops: int = None, similarity_threshold: float = None) -> Tuple[str, float, List[str]]:
        if max_hops is None:
            max_hops = settings.MAX_HOPS
        if similarity_threshold is None:
            similarity_threshold = settings.SIMILARITY_THRESHOLD
            
        question_embedding = self.nlp_processor.get_embedding(question)
        question_entities = self.nlp_processor.extract_entities(question)
        
        relevant_nodes = self.db.find_similar_nodes(question_embedding, similarity_threshold)
        
        if not relevant_nodes:
            return "I'm sorry, I couldn't find any relevant information to answer your question.", 0.0, []
        
        subgraph = self.db.build_subgraph(relevant_nodes, max_hops)
        
        best_path = self._find_best_path(subgraph, question_entities)
        
        if not best_path:
            return "I'm sorry, I couldn't find a clear path to answer your question.", 0.0, []
        
        answer = self._generate_answer(best_path)
        confidence = self._calculate_confidence(best_path, question_embedding)
        
        return answer, confidence, best_path
    
    def _find_best_path(self, graph: nx.Graph, entities: List[str]) -> List[str]:
        if not graph.nodes():
            return []
            
        start_node = max(graph.nodes(), key=lambda x: graph.degree(x))
        paths = nx.single_source_shortest_path(graph, start_node)
        
        best_path = []
        max_entity_matches = 0
        
        for target, path in paths.items():
            entity_matches = sum(1 for entity in entities if entity in str(path))
            if entity_matches > max_entity_matches:
                max_entity_matches = entity_matches
                best_path = path
                
        return best_path
    
    def _generate_answer(self, path: List[str]) -> str:
        if not path:
            return "I couldn't find a clear answer."
            
        return f"Based on the information in the knowledge graph, {' -> '.join(path)}"
    
    def _calculate_confidence(self, path: List[str], question_embedding: torch.Tensor) -> float:
        if not path:
            return 0.0
            
        return 0.8 