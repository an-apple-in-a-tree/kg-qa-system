import pytest
from src.qa_system import QASystem
from src.nlp.processor import NLPProcessor
from src.database.neo4j_client import Neo4jClient

@pytest.fixture
def qa_system():
    system = QASystem()
    yield system
    system.close()

@pytest.fixture
def nlp_processor():
    return NLPProcessor()

def test_question_processing(qa_system):
    question = "What is the relationship between John and Mary?"
    result = qa_system.process_question(question)
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1

def test_entity_extraction(nlp_processor):
    text = "John works at Microsoft in Seattle"
    entities = nlp_processor.extract_entities(text)
    
    assert isinstance(entities, list)
    assert len(entities) > 0
    for entity in entities:
        assert "text" in entity
        assert "label" in entity
        assert "start" in entity
        assert "end" in entity

def test_question_classification(nlp_processor):
    questions = {
        "What is the capital of France?": "factual",
        "How does photosynthesis work?": "explanatory",
        "When was the Eiffel Tower built?": "temporal_spatial",
        "Tell me about quantum physics": "general"
    }
    
    for question, expected_type in questions.items():
        result = nlp_processor.process_question(question)
        assert result["question_type"] == expected_type

def test_similarity_calculation(nlp_processor):
    text1 = "The cat sat on the mat"
    text2 = "A cat is sitting on a mat"
    similarity = nlp_processor.calculate_similarity(text1, text2)
    
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1

def test_empty_question(qa_system):
    result = qa_system.process_question("")
    assert result["confidence"] == 0.0
    assert "couldn't identify" in result["answer"].lower()

def test_invalid_question(qa_system):
    result = qa_system.process_question("!@#$%^&*()")
    assert result["confidence"] == 0.0
    assert "couldn't identify" in result["answer"].lower() 