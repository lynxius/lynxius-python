from lynxius.evals.custom_eval import CustomEval
from lynxius_evals.evaluators.llm_based_eval import LLMBasedEval
from lynxius.evals.local.evaluator_local import EvaluatorLocal
from lynxius_evals.models.openai import OpenAIModel
from lynxius_evals.prompts.eval_prompt import EvalPromptTemplate


class CustomEvalLocal(CustomEval, EvaluatorLocal):
    def __init__(
        self,
        label: str,
        prompt_template: str,
        name: str = None,
        href: str = None,
        tags: list[str] = [],
    ):
        super().__init__(label, prompt_template, name, href, tags)

        self.evaluated_results = None

    def evaluate_local(self):
        model = OpenAIModel()
        template = EvalPromptTemplate("CustomEval", self.prompt_template)
        eval = LLMBasedEval(
            model,
            template,
            # TODO: now that we run this locally, we can expose output_map to be set
            # manually.
            output_map={"correct": True, "incorrect": False},
            output_default=False,
        )

        variables = []
        for sample in self.samples:
            # Merge contexts and variables to be formatted in the prompt
            variables.append(sample[0] | {"contexts": sample[1]})

        self.evaluated_results = eval.evaluate(variables)

    def get_url(self):
        return "/evals/store/custom_eval/"

    def get_request_body(self):
        if self.evaluated_results is None:
            raise Exception(
                "Call evaluator.evaluate_local() first and then store the evaluator output"
            )

        body = {
            "label": self.label,
            "href": self.href,
            "tags": self.tags,
            "prompt_template": self.prompt_template,
            "name_override": self.name_override,
            "version": "1",
            "data": [
                {
                    "llm_input": result["llm_input"],
                    "llm_output": result["llm_output"],
                    "variables": self.samples[i][0],
                    "score": str(result["score"]),
                    "contexts": [c.__dict__ for c in result["contexts"]],
                }
                for i, result in enumerate(self.evaluated_results)
            ],
        }

        return body
