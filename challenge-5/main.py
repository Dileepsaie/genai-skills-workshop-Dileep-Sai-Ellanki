import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

import rag_service

app = FastAPI(title="Alaska Department of Snow - Online Agent")

class ChatRequest(BaseModel):
    message: str
    top_k: int = 10
    session_id: str | None = None

class ChatResponse(BaseModel):
    session_id: str
    blocked: bool
    answer: str
    valid: bool | None = None
    issues: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    out = rag_service.guarded_rag_chat(req.message, top_k=req.top_k, session_id=sid)
    return {
        "session_id": out["session_id"],
        "blocked": bool(out.get("blocked", False)),
        "answer": out["answer"],
        "valid": out.get("valid", True),
        "issues": out.get("issues", ""),
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return rag_service.html_page()
