import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class KnowledgePyramid:
    def __init__(self, text):
        self.raw_text = text
        # Layer 1: Sliding Window (Approx 2000 chars per "page", 1 page overlap)
        self.windows = self._create_sliding_window(text)
        self.pyramid = []
        self._build_pyramid()

    def _create_sliding_window(self, text, page_size=2000, overlap=1000):
        windows = []
        for i in range(0, len(text), page_size - overlap):
            windows.append(text[i:i + page_size])
        return windows

    def _build_pyramid(self):
        for chunk in self.windows:
            layer = {
                "raw": chunk,
                "summary": chunk[:200] + "...",  # Placeholder Summarization
                "theme": self._get_theme(chunk),   # Rule-based Theme
                "distilled": self._extract_keywords(chunk) # Knowledge distillation
            }
            self.pyramid.append(layer)

    def _get_theme(self, text):
        themes = {"tech": ["code", "ai", "software"], "finance": ["money", "market", "stock"]}
        for theme, keywords in themes.items():
            if any(k in text.lower() for k in keywords): return theme
        return "General"

    def _extract_keywords(self, text):
        # Mocking keyword extraction
        return list(set(re.findall(r'\b\w{6,}\b', text)))[:5]

    def query(self, user_query):
        # Simple TF-IDF similarity across all pyramid levels
        all_contents = []
        for p in self.pyramid:
            all_contents.extend([p['raw'], p['summary'], p['theme'], " ".join(p['distilled'])])
        
        vectorizer = TfidfVectorizer().fit_transform(all_contents + [user_query])
        vectors = vectorizer.toarray()
        cosine_sims = cosine_similarity([vectors[-1]], vectors[:-1])[0]
        
        best_idx = cosine_sims.argmax()
        return all_contents[best_idx]

# Example Usage
if __name__ == "__main__":
    sample_text = "Artificial Intelligence is transforming software development. AI engineers use Python to build RAG systems." * 50
    engine = KnowledgePyramid(sample_text)
    print(f"Query Result: {engine.query('How is AI used in software?')}")