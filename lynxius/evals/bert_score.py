from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk


class BertScore(Evaluator):
    def __init__(
        self,
        label: str,
        level: str = "word",
        presence_threshold: float = 0.65,
        href: str = None,
        tags: list[str] = [],
    ):
        levels = ["word", "sentence"]
        if level not in levels:
            raise ValueError(f"Level must be one of the following: {levels}")

        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.level = level
        self.presence_threshold = presence_threshold
        self.samples = []

    def add_trace(self, reference: str, output: str, context: list[ContextChunk] = []):
        if not reference or not output:
            raise ValueError("Both reference and output must be provided")

        self.samples.append((reference, output, context))

    def get_url(self):
        return "/evals/run/bert_score/"

    def get_request_body(self):
        body = {
            "label": self.label,
            "href": self.href,
            "tags": self.tags,
            "level": self.level,
            "presence_threshold": self.presence_threshold,
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
