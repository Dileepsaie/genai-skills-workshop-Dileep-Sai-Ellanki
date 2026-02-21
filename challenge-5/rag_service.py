import os
import json
import re
import datetime
from typing import List, Dict, Any

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel
from google.cloud import bigquery


# ========================
# Config / Environment
# ========================
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

RAG_TABLE_ID = os.environ.get("RAG_TABLE_ID")  # e.g. myproj.ads_rag.chunks
LOG_TABLE_ID = os.environ.get("LOG_TABLE_ID")  # e.g. myproj.ads_logs.chat_logs

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-005")

if not PROJECT_ID:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT not set.")
if not RAG_TABLE_ID:
    raise RuntimeError("RAG_TABLE_ID env var not set (e.g. PROJECT.ads_rag.chunks).")
if not LOG_TABLE_ID:
    raise RuntimeError("LOG_TABLE_ID env var not set (e.g. PROJECT.ads_logs.chat_logs).")


# ========================
# Clients (Vertex + BQ)
# ========================
vertexai.init(project=PROJECT_ID, location=LOCATION)
bq = bigquery.Client(project=PROJECT_ID)

gemini = GenerativeModel(MODEL_NAME)
embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)

GEN_ANS = GenerationConfig(temperature=0.2, top_p=0.95, max_output_tokens=768)

CITE_RE = re.compile(r"\[[^\]]+#\d+\]")


# ========================
# Helpers
# ========================
def safe_text(resp) -> str:
    """Safely extract text from Gemini response without crashing on .text."""
    try:
        t = getattr(resp, "text", None)
        if t:
            return t.strip()
    except Exception:
        pass

    try:
        cands = getattr(resp, "candidates", []) or []
        if not cands:
            return ""
        parts = getattr(getattr(cands[0], "content", None), "parts", None) or []
        return "".join([getattr(p, "text", "") for p in parts]).strip()
    except Exception:
        return ""


def embed_texts(texts: List[str]) -> List[List[float]]:
    embs = embed_model.get_embeddings(texts)
    return [e.values for e in embs]


def retrieve_top_chunks(query: str, top_k: int = 10):
    q_vec = embed_texts([query])[0]
    sql = f"""
    SELECT base.doc_uri, base.doc_path, base.chunk_id, base.chunk_text, distance
    FROM VECTOR_SEARCH(
      TABLE `{RAG_TABLE_ID}`,
      'embedding',
      (SELECT @qvec AS embedding),
      top_k => @topk
    )
    """
    job = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("qvec", "FLOAT64", q_vec),
                bigquery.ScalarQueryParameter("topk", "INT64", top_k),
            ]
        ),
    )
    return list(job.result())


def build_context(hits) -> str:
    lines = []
    for r in hits:
        cite = f"[{r.doc_path}#{int(r.chunk_id)}]"
        lines.append(f"{cite}\n{r.chunk_text}")
    return "\n\n---\n\n".join(lines)


# ========================
# RAG Answer
# ========================
def rag_answer(question: str, top_k: int = 10) -> Dict[str, Any]:
    hits = retrieve_top_chunks(question, top_k=top_k)
    context = build_context(hits)

    prompt = f"""
You are the Alaska Department of Snow online assistant.
Use ONLY the context below.

Requirements:
- If the answer is not in the context, say exactly:
  "I don't have enough information in the provided documents."
- Add inline citations like [doc_path#chunk_id] after each key fact.
- Write 2–6 sentences. End with a complete sentence.

User question:
{question}

Context:
{context}

Answer:
""".strip()

    resp = gemini.generate_content(prompt, generation_config=GEN_ANS)
    ans = safe_text(resp)

    # Retry once if empty/truncated
    if (not ans) or (ans and ans[-1] not in ".!?"):
        resp2 = gemini.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.2, max_output_tokens=1024),
        )
        ans2 = safe_text(resp2)
        if ans2:
            ans = ans2

    retrieved_meta = [
        {
            "doc_path": r.doc_path,
            "chunk_id": int(r.chunk_id),
            "distance": float(r.distance),
        }
        for r in hits
    ]

    return {"answer": ans, "retrieved": retrieved_meta}


# ========================
# Guardrails (FIXED)
# ========================
def prompt_filter(user_query: str) -> Dict[str, str]:
    """
    Deterministic safety gate so we NEVER block everything
    due to JSON parsing issues from an LLM.

    BLOCK only clear unsafe / illegal intents.
    Otherwise ALLOW.
    """
    q = (user_query or "").lower().strip()

    blocked_phrases = [
        # weapons/explosives/violence
        "how to build a bomb", "make a bomb", "explosive", "molotov",
        "how to kill", "kill someone", "murder", "shoot", "stab",
        # self-harm
        "suicide", "kill myself", "self harm",
        # illegal / sabotage
        "sabotage", "disable a plow", "destroy", "poison", "ricin",
        "steal", "hack", "bypass", "jailbreak",
    ]

    if any(p in q for p in blocked_phrases):
        return {"decision": "BLOCK", "reason": "Unsafe or illegal request."}

    return {"decision": "ALLOW", "reason": "Looks safe."}


def validate_answer(answer: str) -> Dict[str, Any]:
    issues = []
    if not answer or len(answer.strip()) < 5:
        issues.append("empty_or_too_short")

    # Require citations unless explicitly no-info response
    if answer.strip() != "I don't have enough information in the provided documents.":
        if not CITE_RE.search(answer):
            issues.append("missing_citations")

    return {"valid": len(issues) == 0, "issues": ", ".join(issues) if issues else ""}


# ========================
# Logging
# ========================
def log_chat(
    session_id: str,
    user_query: str,
    gate: Dict[str, str],
    top_k: int,
    retrieved: list,
    answer: str,
    validation: dict,
):
    row = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session_id": session_id,
        "user_query": user_query,
        "prompt_allowed": (gate["decision"] == "ALLOW"),
        "prompt_reason": gate.get("reason", ""),
        "top_k": int(top_k),
        "retrieved": json.dumps(retrieved),
        "answer": answer or "",
        "answer_valid": bool(validation["valid"]),
        "answer_issues": validation.get("issues", ""),
    }
    bq.insert_rows_json(LOG_TABLE_ID, [row])


# ========================
# Main Chat Orchestrator
# ========================
def guarded_rag_chat(user_query: str, top_k: int = 10, session_id: str = "") -> Dict[str, Any]:
    gate = prompt_filter(user_query)

    if gate["decision"] == "BLOCK":
        answer = "Sorry—I can’t help with that request."
        validation = {"valid": True, "issues": ""}
        log_chat(session_id, user_query, gate, top_k, [], answer, validation)
        return {
            "session_id": session_id,
            "answer": answer,
            "blocked": True,
            "valid": True,
            "issues": "",
        }

    out = rag_answer(user_query, top_k=top_k)
    answer = out["answer"]
    validation = validate_answer(answer)

    # One retry if missing citations
    if (not validation["valid"]) and ("missing_citations" in validation["issues"]):
        out = rag_answer(user_query + " (Include citations like [doc_path#chunk_id].)", top_k=top_k)
        answer = out["answer"]
        validation = validate_answer(answer)

    log_chat(session_id, user_query, gate, top_k, out["retrieved"], answer, validation)

    return {
        "session_id": session_id,
        "answer": answer,
        "blocked": False,
        "valid": validation["valid"],
        "issues": validation["issues"],
    }


# ========================
# HTML Page
# ========================
def html_page() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Alaska Department of Snow — Online Agent</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 30px auto; padding: 0 12px; }
    .box { border: 1px solid #ddd; border-radius: 12px; padding: 14px; }
    .row { margin: 10px 0; }
    textarea { width: 100%; height: 70px; }
    button { padding: 10px 14px; border-radius: 10px; border: 1px solid #333; background: #111; color: #fff; cursor: pointer; }
    pre { white-space: pre-wrap; background: #f7f7f7; padding: 12px; border-radius: 10px; }
  </style>
</head>
<body>
  <h1>Alaska Department of Snow — Online Agent</h1>
  <div class="box">
    <div class="row">
      <label>Message</label><br/>
      <textarea id="msg" placeholder="Ask a question..."></textarea>
    </div>
    <div class="row">
      <button onclick="send()">Send</button>
    </div>
    <div class="row">
      <label>Answer</label>
      <pre id="out"></pre>
    </div>
  </div>

<script>
async function send(){
  const msg = document.getElementById("msg").value;
  document.getElementById("out").textContent = "Thinking...";
  const r = await fetch("/chat", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({message: msg, top_k: 10})
  });
  const j = await r.json();
  document.getElementById("out").textContent = j.answer;
}
</script>
</body>
</html>
""".strip()
