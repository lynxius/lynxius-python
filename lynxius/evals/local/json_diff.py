from lynxius.evals.json_diff import JsonDiff
from lynxius_evals.evaluators.json_diff_evaluator import JsonDiffEval
from lynxius.evals.local.evaluator_local import EvaluatorLocal


class JsonDiffLocal(JsonDiff, EvaluatorLocal):
    def __init__(self, label: str, href: str = None, tags: list[str] = []):
        super().__init__(label, href, tags)

        self.evaluated_results = None

    def evaluate(self):
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

    def get_url(self):
        return "/evals/store/json_diff_eval/"

    def get_request_body(self):
        if self.evaluated_results is None:
            raise Exception(
                "Call evaluator.evaluate() first and then store the evaluator output"
            )

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

        return body
