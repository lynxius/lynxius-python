import json
from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk
from lynxius_evals.evaluators.semantic_similarity_eval import SemanticSimilarityEval
from lynxius_evals.models.openai import OpenAIModel


class SemanticSimilarity(Evaluator):
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

    def add_trace(
        self,
        reference: str,
        output: str,
        context: list[ContextChunk] = [],
        trace_uuid: str | None = None,
    ):
        if not reference or not output:
            raise ValueError("Both reference and output must be provided")

        self.samples.append(
            {
                "reference": reference,
                "output": output,
                "contexts": context,
                "trace_uuid": trace_uuid,
            }
        )

    def get_url(self, run_local: bool = False):
        return (
            "/evals/store/semantic_similarity/"
            if run_local
            else "/evals/run/semantic_similarity/"
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
                        "reference": result["reference"],
                        "output": result["output"],
                        "similarity": result["similarity"],
                        "contexts": [c.__dict__ for c in result["contexts"]],
                        "trace_uuid": result["trace_uuid"],
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
                        "reference": item["reference"],
                        "output": item["output"],
                        "contexts": [c.__dict__ for c in item["contexts"]],
                        "trace_uuid": item["trace_uuid"],
                    }
                    for item in self.samples
                ],
            }

        return body

    def evaluate_local(self):
        model = OpenAIModel()
        eval = SemanticSimilarityEval(model)

        variables = []
        for sample in self.samples:
            variables.append(
                {
                    "reference": sample["reference"],
                    "output": sample["output"],
                    "contexts": sample["contexts"],
                    "trace_uuid": sample["trace_uuid"],
                }
            )

        self.evaluated_results = eval.evaluate(variables)

        super().evaluate_local()

    def get_merge_id(self) -> int:
        return hash(
            json.dumps(
                {
                    "class": str(self.__class__),
                    "label": self.label,
                    "href": self.href,
                    "tags": self.tags,
                    "baseline_project_uuid": self.baseline_project_uuid,
                    "baseline_eval_run_label": self.baseline_eval_run_label,
                }
            )
        )
