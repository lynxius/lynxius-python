from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk


class JsonNumeric(Evaluator):
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

            def traverse(obj):
                if isinstance(obj, dict):
                    for item in obj.values():
                        traverse(item)
                elif (
                    isinstance(obj, list)
                    or isinstance(obj, str)
                    or isinstance(obj, bool)
                ):
                    raise ValueError("Weights object can contain only floats or ints")
                elif isinstance(obj, float) or isinstance(obj, int):
                    if obj < 0 or obj > 1:
                        raise ValueError("Weights should be in the range of [0.0, 1.0]")

            traverse(weights)

        self.samples.append((reference, output, weights, context))

    def get_url(self):
        return "/evals/run/json_numeric_eval/"

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
