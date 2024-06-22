import json
from lynxius.evals.context_precision import ContextPrecision
from lynxius_evals.evaluators.context_precision_eval import ContextPrecisionEval
from lynxius.evals.local.evaluator_local import EvaluatorLocal
from lynxius_evals.models.openai import OpenAIModel
from lynxius_evals.prompts.context_precision_prompt import CONTEXT_PRECISION_TEMPLATE


class ContextPrecisionLocal(ContextPrecision, EvaluatorLocal):
    def __init__(self, label: str, href: str = None, tags: list[str] = []):
        super().__init__(label, href, tags)

        self.evaluated_results = None

    def evaluate_local(self):
        model = OpenAIModel(response_format="json_object")
        eval = ContextPrecisionEval(
            model,
            CONTEXT_PRECISION_TEMPLATE,
        )

        variables = []
        for sample in self.samples:
            variables.append(
                {
                    "query": sample[0],
                    "reference": sample[1],
                    "contexts": [c.__dict__ for c in sample[2]],
                }
            )

        self.evaluated_results = eval.evaluate(variables)

    def get_url(self):
        return "/evals/store/context_precision_eval/"

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
                    "query": result["query"],
                    "reference": result["reference"],
                    "llm_input": result["llm_input"],
                    "llm_output": json.loads(result["llm_output"]),
                    "score": result["score"],
                    "contexts": result["contexts"],
                }
                for result in self.evaluated_results
            ],
        }

        return body
