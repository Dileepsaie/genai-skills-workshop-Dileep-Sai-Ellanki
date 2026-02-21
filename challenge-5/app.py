import re
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

ALLOWED_CATEGORIES = {"Employment", "General Information", "Emergency Services", "Tax Related"}

GEN_CONFIG_DETERMINISTIC = GenerationConfig(temperature=0.0, top_p=1.0, max_output_tokens=256)
GEN_CONFIG_POSTS = GenerationConfig(temperature=0.2, top_p=0.95, max_output_tokens=256)

def init_model(project_id: str, location: str = "us-central1", model_name: str = "gemini-2.5-flash"):
    vertexai.init(project=project_id, location=location)
    return GenerativeModel(model_name)

def _safe_text(resp) -> str:
    """
    Returns response text if present; otherwise returns "".
    NEVER raises on empty/blocked candidates.
    """
    # 1) Try resp.text but protect against ValueError
    try:
        t = getattr(resp, "text", None)
        if t:
            return t.strip()
    except Exception:
        pass

    # 2) Try candidates[0].content.parts[].text
    try:
        cands = getattr(resp, "candidates", []) or []
        if not cands:
            return ""
        parts = getattr(getattr(cands[0], "content", None), "parts", None) or []
        out = "".join([getattr(p, "text", "") for p in parts]).strip()
        return out
    except Exception:
        return ""

def _generate_text_with_retry(model, prompt: str, generation_config: GenerationConfig) -> str:
    """
    Calls model.generate_content and returns safe text.
    Retries once with softened wording if first return is empty.
    """
    resp = model.generate_content(prompt, generation_config=generation_config)
    out = _safe_text(resp)
    if out:
        return out

    # Retry once with slightly safer phrasing
    softened = (
        prompt.replace("Emergency alert", "Public notice")
              .replace("boil-water", "water advisory")
              .replace("gas", "odor")
              .replace("fire", "urgent situation")
    )
    resp2 = model.generate_content(softened, generation_config=generation_config)
    return _safe_text(resp2)

def classify_question(model, question: str) -> str:
    prompt = f"""
Classify the question into EXACTLY one of these labels:
Employment
General Information
Emergency Services
Tax Related

Rules:
- Output ONLY the exact label text above.
- No extra words, punctuation, or explanation.

Question: {question}
Output:
""".strip()

    out = _generate_text_with_retry(model, prompt, GEN_CONFIG_DETERMINISTIC)
    out = out.replace(".", "").strip()

    # normalize common shortenings
    out = {"Emergency": "Emergency Services", "Tax": "Tax Related", "Taxes": "Tax Related"}.get(out, out)

    if out not in ALLOWED_CATEGORIES:
        raise ValueError(f"Unexpected category: {out!r}")
    return out

def generate_announcement(model, topic: str) -> str:
    prompt = f"""
Write ONE professional government social media post.

Rules:
- Max 200 characters
- MUST include the exact phrase: "Check for updates"
- Output ONLY the post text

Topic: {topic}
Post:
""".strip()

    post = _generate_text_with_retry(model, prompt, GEN_CONFIG_POSTS)
    post = re.sub(r"\s+", " ", post).strip()

    # Hard enforcement for deterministic tests
    if "check for updates" not in post.lower():
        if post and not post.endswith((".", "!", "?")):
            post += "."
        post = f"{post} Check for updates."

    return post[:200]
