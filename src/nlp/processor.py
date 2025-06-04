import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from config.config import settings

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load(settings.SPACY_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModel.from_pretrained(settings.MODEL_NAME)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities
    
    def get_embeddings(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)
        similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity[0][0])
    
    def process_question(self, question: str) -> Dict[str, Any]:
        entities = self.extract_entities(question)
        question_type = self._classify_question_type(question)
        return {
            "entities": entities,
            "question_type": question_type,
            "embedding": self.get_embeddings(question)
        }
    
    def _classify_question_type(self, question: str) -> str:
        question = question.lower()
        if any(w in question for w in ["what", "which", "who"]):
            return "factual"
        elif any(w in question for w in ["how", "why"]):
            return "explanatory"
        elif any(w in question for w in ["when", "where"]):
            return "temporal_spatial"
        else:
            return "general" 