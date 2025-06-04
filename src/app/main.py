from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.qa_system import QASystem
from config.config import settings

app = FastAPI(
    title="Knowledge Graph QA System",
    description="A question-answering system based on knowledge graphs",
    version="1.0.0"
)

qa_system = QASystem()

class Question(BaseModel):
    text: str
    max_hops: Optional[int] = settings.MAX_HOPS
    similarity_threshold: Optional[float] = settings.SIMILARITY_THRESHOLD

class Answer(BaseModel):
    answer: str
    confidence: float
    path: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to the Knowledge Graph QA System"}

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    try:
        answer, confidence, path = qa_system.answer_question(
            question.text,
            max_hops=question.max_hops,
            similarity_threshold=question.similarity_threshold
        )
        return Answer(answer=answer, confidence=confidence, path=path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 