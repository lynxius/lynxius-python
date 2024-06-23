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
        run_local: bool = False,
    ):
        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.samples = []
        self.run_local = run_local
        self.evaluated_results = None

    def add_trace(
        self, query: str, reference: str, output: str, context: list[ContextChunk] = []
    ):
        if not query or not reference or not output:
            raise ValueError("Query, reference and output must all be provided")

        self.samples.append((query, reference, output, context))

    def get_url(self):
        return (
            "/evals/store/answer_correctness/"
            if self.run_local
            else "/evals/run/answer_correctness/"
        )

    def get_request_body(self):
        if self.run_local and self.evaluated_results is None:
            raise Exception(
                "Call evaluator.evaluate_local() first and then store the evaluator output"
            )

        if self.run_local:
            body = {
                "label": self.label,
                "href": self.href,
                "tags": self.tags,
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
                        "query": item[0],
                        "reference": item[1],
                        "output": item[2],
                        "contexts": [c.__dict__ for c in item[3]],
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
                    "query": sample[0],
                    "reference": sample[1],
                    "output": sample[2],
                    "contexts": sample[3],
                }
            )

        self.evaluated_results = eval.evaluate(variables)
