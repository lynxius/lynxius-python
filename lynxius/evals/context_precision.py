import json

from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk
from lynxius_evals.evaluators.context_precision_eval import ContextPrecisionEval
from lynxius_evals.models.openai import OpenAIModel
from lynxius_evals.prompts.context_precision_prompt import CONTEXT_PRECISION_TEMPLATE


class ContextPrecision(Evaluator):
    def __init__(
        self,
        label: str,
        href: str = None,
        tags: list[str] = [],
        baseline_project_uuid: str = None,
        baseline_eval_run_label: str = None,
    ):
        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.baseline_project_uuid = baseline_project_uuid
        self.baseline_eval_run_label = baseline_eval_run_label
        self.samples = []
        self.evaluated_results = None

    def add_trace(self, query: str, reference: str, context: list[ContextChunk] = []):
        if not query or not reference:
            raise ValueError("Query and reference output must be provided")

        self.samples.append((query, reference, context))

    def get_url(self, run_local: bool = False):
        return (
            "/evals/store/context_precision_eval/"
            if run_local
            else "/evals/run/context_precision_eval/"
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
                "baseline_project_uuid": self.baseline_project_uuid,
                "baseline_eval_run_label": self.baseline_eval_run_label,
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
        else:
            body = {
                "label": self.label,
                "href": self.href,
                "tags": self.tags,
                "baseline_project_uuid": self.baseline_project_uuid,
                "baseline_eval_run_label": self.baseline_eval_run_label,
                "data": [
                    {
                        "query": item[0],
                        "reference": item[1],
                        "contexts": [c.__dict__ for c in item[2]],
                    }
                    for item in self.samples
                ],
            }

        return body

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
