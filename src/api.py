from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from qa_system import QASystem
from config.config import settings

app = FastAPI(
    title="Knowledge Graph QA System",
    description="A question-answering system based on knowledge graphs",
    version="1.0.0"
)

qa_system = QASystem()

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    confidence: float
    source_entities: Optional[list] = None

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question) -> Dict[str, Any]:
    try:
        result = qa_system.process_question(question.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    ) 