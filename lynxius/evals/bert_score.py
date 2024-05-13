from lynxius.evals.evaluator import Evaluator


class BertScore(Evaluator):
    def __init__(
        self, title: str, level: str = "word", presence_threshold: float = 0.65
    ):
        levels = ["word", "sentence"]
        if level not in levels:
            raise ValueError(f"Level must be one of the following: {levels}")

        self.title = title
        self.level = level
        self.presence_threshold = presence_threshold
        self.samples = []

    def add_trace(self, reference: str, output: str):
        if not reference or not output:
            raise ValueError("Both reference and output must be provided")

        self.samples.append((reference, output))

    def get_url(self):
        return "/api/evals/run/bert_score/"

    def get_request_body(self):
        body = {
            "title": self.title,
            "level": self.level,
            "presence_threshold": self.presence_threshold,
            "data": [
                {"reference": item[0], "output": item[1], "contexts": []} for item in self.samples
            ],
        }

        return body
