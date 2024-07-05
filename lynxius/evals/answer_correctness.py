import json

from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk
from lynxius_evals.evaluators.answer_correctness_eval import AnswerCorrectnessEval
from lynxius_evals.models.openai import OpenAIModel
from lynxius_evals.prompts.answer_correctness_prompt import ANSWER_CORRECTNESS_TEMPLATE


class AnswerCorrectness(Evaluator):
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

    def add_trace(
        self,
        query: str,
        reference: str,
        output: str,
        context: list[ContextChunk] = [],
        trace_uuid: str | None = None,
    ):
        if not query or not reference or not output:
            raise ValueError("Query, reference and output must all be provided")

        self.samples.append(
            {
                "query": query,
                "reference": reference,
                "output": output,
                "contexts": context,
                "trace_uuid": trace_uuid,
            }
        )

    def get_url(self, run_local: bool = False):
        return (
            "/evals/store/answer_correctness/"
            if run_local
            else "/evals/run/answer_correctness/"
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
                        "output": result["output"],
                        "llm_input": result["llm_input"],
                        "llm_output": json.loads(result["llm_output"]),
                        "contexts": [c.__dict__ for c in result["contexts"]],
                        "score": result["score"],
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
                        "query": item["query"],
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
        model = OpenAIModel(response_format="json_object")
        eval = AnswerCorrectnessEval(
            model,
            ANSWER_CORRECTNESS_TEMPLATE,
        )

        variables = []
        for sample in self.samples:
            variables.append(
                {
                    "query": sample["query"],
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
