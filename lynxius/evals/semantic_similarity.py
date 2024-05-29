from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk


class SemanticSimilarity(Evaluator):
    def __init__(self, label: str, href: str = None, tags: list[str] = []):
        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.samples = []

    def add_trace(self, reference: str, output: str, context: list[ContextChunk] = []):
        if not reference or not output:
            raise ValueError("Both reference and output must be provided")

        self.samples.append((reference, output, context))

    def get_url(self):
        return "/evals/run/semantic_similarity/"

    def get_request_body(self):
        body = {
            "label": self.label,
            "href": self.href,
            "tags": self.tags,
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
