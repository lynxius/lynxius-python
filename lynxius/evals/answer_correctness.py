from lynxius.evals.evaluator import Evaluator


class AnswerCorrectness(Evaluator):
    def __init__(self, title: str):
        self.title = title
        self.samples = []

    def add_trace(self, query: str, reference: str, output: str):
        if not query or not reference or not output:
            raise ValueError("Query, reference and output must all be provided")

        self.samples.append((query, reference, output))

    def get_url(self):
        return "/api/evals/run/answer_correctness/"

    def get_request_body(self):
        body = {
            "title": self.title,
            "data": [
                {"query": item[0], "reference": item[1], "output": item[2], "contexts": []}
                for item in self.samples
            ],
        }

        return body
