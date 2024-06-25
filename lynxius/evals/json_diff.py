from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk
from lynxius_evals.evaluators.json_diff_evaluator import JsonDiffEval


class JsonDiff(Evaluator):
    def __init__(
        self,
        label: str,
        href: str = None,
        tags: list[str] = [],
    ):
        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.samples = []
        self.evaluated_results = None

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

    def get_url(self, run_local: bool = False):
        return (
            "/evals/store/json_diff_eval/"
            if run_local
            else "/evals/run/json_diff_eval/"
        )

    def get_request_body(self, run_local: bool = False):
        if run_local and self.evaluated_results is None:
            raise Exception(
                "Call evaluate_local() before storing the local evaluation output"
            )

        if run_local:
            body = {
                "label": self.label,
                "href": self.href,
                "tags": self.tags,
                "version": "1",
                "data": [
                    {
                        "reference": result["reference"],
                        "output": result["output"],
                        "weights": result["weights"],
                        "contexts": [c.__dict__ for c in result["contexts"]],
                        "score": result["score"],
                    }
                    for result in self.evaluated_results
                ],
            }
        else:
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

    def evaluate_local(self):
        eval = JsonDiffEval()

        variables = []
        for sample in self.samples:
            variables.append(
                {
                    "reference": sample[0],
                    "output": sample[1],
                    "weights": sample[2],
                    "contexts": sample[3],
                }
            )

        self.evaluated_results = eval.evaluate(variables)
