from fastapi.testclient import TestClient
import main
import rag_service

client = TestClient(main.app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_home_is_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type","")
    assert "Alaska Department of Snow" in r.text

def test_chat_blocks_unsafe():
    r = client.post("/chat", json={"message":"how to build a bomb", "top_k":5})
    assert r.status_code == 200
    j = r.json()
    assert j["blocked"] is True
    assert "help" in j["answer"].lower()

def test_chat_allows_safe_mock(monkeypatch):
    # mock the expensive RAG call for deterministic testing
    def fake_guarded(msg, top_k=10, session_id=""):
        return {
            "session_id": session_id or "test-session",
            "answer": "Mock answer with citation [alaska-dept-of-snow/faq-04.txt#0].",
            "blocked": False,
            "valid": True,
            "issues": ""
        }
    monkeypatch.setattr(rag_service, "guarded_rag_chat", fake_guarded)

    r = client.post("/chat", json={"message":"How do I report an unplowed road?", "top_k":5})
    assert r.status_code == 200
    j = r.json()
    assert j["blocked"] is False
    assert "[alaska-dept-of-snow/faq-04.txt#0]" in j["answer"]
