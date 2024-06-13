from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk


class JsonDiff(Evaluator):
    def __init__(self, label: str, href: str = None, tags: list[str] = []):
        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.samples = []

    def add_trace(
        self,
        reference: dict,
        output: dict,
        weights: dict = {},
        context: list[ContextChunk] = [],
    ):
        if not reference or not output:
            raise ValueError("Both reference and output must be provided")

        if weights is not None:

            def sum_dict(d):
                total = 0
                if isinstance(d, list) or isinstance(d, str) or isinstance(d, bool):
                    raise ValueError(f"Weights object can contain only floats or ints, not: {d}")
                elif isinstance(d, (int, float)):
                    total += d
                elif isinstance(d, dict):
                    for key, value in d.items():
                        parent_sum = sum_dict(value)
                        total += sum_dict(value)
                        if not (0.0 <= parent_sum <= 1.0):
                            raise ValueError(f'The sum of the weights within key "{key}" is not within [0.0, 1.0], but is: {parent_sum}')
                return total

            sum_dict(weights)

        self.samples.append((reference, output, weights, context))

    def get_url(self):
        return "/evals/run/json_diff_eval/"

    def get_request_body(self):
        body = {
            "label": self.label,
            "href": self.href,
            "tags": self.tags,
            "data": [
                {
                    "reference": item[0],
                    "output": item[1],
                    "weights": item[2],
                    "contexts": [c.__dict__ for c in item[3]],
                }
                for item in self.samples
            ],
        }

        return body
