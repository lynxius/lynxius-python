from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk


class SemanticSimilarity(Evaluator):
    def __init__(self, title: str):
        self.title = title
        self.samples = []

    def add_trace(self, reference: str, output: str, context: list[ContextChunk] = []):
        if not reference or not output:
            raise ValueError("Both reference and output must be provided")

        self.samples.append((reference, output, context))

    def get_url(self):
        return "/api/evals/run/semantic_similarity/"

    def get_request_body(self):
        body = {
            "title": self.title,
            "data": [
                {
                    "reference": item[0],
                    "output": item[1],
                    "contexts": [c.__dict__ for c in item[2]],
                }
                for item in self.samples
            ],
        }

        return body
