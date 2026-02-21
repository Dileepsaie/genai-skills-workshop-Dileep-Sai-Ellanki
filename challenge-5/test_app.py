import re
import app

class FakeResponse:
    def __init__(self, text):
        self._text = text
    @property
    def text(self):
        return self._text

class FakeModel:
    """
    Fake Gemini model for unit tests.
    IMPORTANT: classify based on the *question* section only,
    because the prompt contains the label list (including words like 'Emergency').
    """
    def generate_content(self, prompt, generation_config=None):
        p = prompt.lower()

        # ---- Extract the question text if present ----
        # Works for prompts like:
        # "Question: ...\nOutput:"
        m = re.search(r"question:\s*(.+?)\n\s*output:", prompt, flags=re.IGNORECASE | re.DOTALL)
        question_only = (m.group(1).strip().lower() if m else "")

        # ---- Classification ----
        if "classify the question" in p:
            q = question_only

            if "tax" in q or "property taxes" in q:
                return FakeResponse("Tax Related")
            if "apply" in q and ("job" in q or "position" in q):
                return FakeResponse("Employment")
            if "library" in q or "hours" in q or "city hall" in q or "open" in q:
                return FakeResponse("General Information")
            if "fire" in q or "smell gas" in q or "emergency" in q:
                return FakeResponse("Emergency Services")

            return FakeResponse("General Information")

        # ---- Announcement generation ----
        if "social media post" in p or "announcement" in p:
            return FakeResponse("City update: Schools closed tomorrow. Check for updates.")

        return FakeResponse("General Information")


# Use fake model for all tests
model = FakeModel()

def test_classify_employment():
    assert app.classify_question(model, "How do I apply for a job with the city?") == "Employment"

def test_classify_general():
    assert app.classify_question(model, "What are the library hours on Saturday?") == "General Information"

def test_classify_emergency():
    assert app.classify_question(model, "There is a fire on my street—who do I call?") == "Emergency Services"

def test_classify_tax():
    assert app.classify_question(model, "When are property taxes due?") == "Tax Related"

def test_announcement_rules():
    post = app.generate_announcement(model, "School closing tomorrow due to snow. Include next steps.")
    assert len(post) <= 200
    assert any(k in post.lower() for k in ["visit","check","call","follow","updates"])
