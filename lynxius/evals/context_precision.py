from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk


class ContextPrecision(Evaluator):
    def __init__(self, title: str):
        self.title = title
        self.samples = []

    def add_trace(self, query: str, reference: str, context: list[ContextChunk] = []):
        if not query or not reference:
            raise ValueError("Query and reference output must be provided")

        self.samples.append((query, reference, context))

    def get_url(self):
        return "/api/evals/run/context_precision_eval/"

    def get_request_body(self):
        body = {
            "title": self.title,
            "data": [
                {
                    "query": item[0],
                    "reference": item[1],
                    "contexts": [c.__dict__ for c in item[2]],
                }
                for item in self.samples
            ],
        }

        return body