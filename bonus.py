class ReasoningRouter:
    """
    A plug-and-play adapter that routes queries to specialized reasoning paths.
    """
    def __init__(self):
        self.routes = {
            "math": self._solve_logic,
            "legal": self._search_compliance,
            "general": self._generate_response
        }

    def _identify_intent(self, query):
        if any(op in query for op in ["+", "-", "/", "*", "calculate"]): return "math"
        if any(word in query for word in ["law", "contract", "regulation"]): return "legal"
        return "general"

    def _solve_logic(self, q): return f"Routing to Symbolic Solver: {q}"
    def _search_compliance(self, q): return f"Routing to RAG Legal Database: {q}"
    def _generate_response(self, q): return f"Routing to Standard LLM: {q}"

    def handle(self, query):
        intent = self._identify_intent(query)
        return self.routes[intent](query)

# Test
router = ReasoningRouter()
print(router.handle("What is 55 * 12?"))