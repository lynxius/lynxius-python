from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk


class AnswerCorrectness(Evaluator):
    def __init__(self, label: str, href: str = None, tags: list[str] = []):
        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.samples = []

    def add_trace(
        self, query: str, reference: str, output: str, context: list[ContextChunk] = []
    ):
        if not query or not reference or not output:
            raise ValueError("Query, reference and output must all be provided")

        self.samples.append((query, reference, output, context))

    def get_url(self):
        return "/evals/run/answer_correctness/"

    def get_request_body(self):
        body = {
            "label": self.label,
            "href": self.href,
            "tags": self.tags,
            "data": [
                {
                    "query": item[0],
                    "reference": item[1],
                    "output": item[2],
                    "contexts": [c.__dict__ for c in item[3]],
                }
                for item in self.samples
            ],
        }

        return body
