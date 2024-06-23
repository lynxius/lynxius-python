import json
from lynxius.evals.answer_correctness import AnswerCorrectness
from lynxius_evals.evaluators.answer_correctness_eval import AnswerCorrectnessEval
from lynxius.evals.local.evaluator_local import EvaluatorLocal
from lynxius_evals.models.openai import OpenAIModel
from lynxius_evals.prompts.answer_correctness_prompt import ANSWER_CORRECTNESS_TEMPLATE


class AnswerCorrectnessLocal(AnswerCorrectness, EvaluatorLocal):
    def __init__(self, label: str, href: str = None, tags: list[str] = []):
        super().__init__(label, href, tags)

        self.evaluated_results = None

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

    def get_url(self):
        return "/evals/store/answer_correctness/"

    def get_request_body(self):
        if self.evaluated_results is None:
            raise Exception(
                "Call evaluator.evaluate_local() first and then store the evaluator output"
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
                    "output": result["output"],
                    "llm_input": result["llm_input"],
                    "llm_output": json.loads(result["llm_output"]),
                    "contexts": [c.__dict__ for c in result["contexts"]],
                    "score": result["score"],
                }
                for result in self.evaluated_results
            ],
        }

        return body
