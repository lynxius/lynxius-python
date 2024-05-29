from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk


class ContextPrecision(Evaluator):
    def __init__(self, label: str, href: str = None, tags: list[str] = []):
        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.samples = []

    def add_trace(self, query: str, reference: str, context: list[ContextChunk] = []):
        if not query or not reference:
            raise ValueError("Query and reference output must be provided")

        self.samples.append((query, reference, context))

    def get_url(self):
        return "/evals/run/context_precision_eval/"

    def get_request_body(self):
        body = {
            "label": self.label,
            "href": self.href,
            "tags": self.tags,
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
