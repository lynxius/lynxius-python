from string import Formatter

from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk
from lynxius_evals.evaluators.llm_based_eval import LLMBasedEval
from lynxius_evals.models.openai import OpenAIModel
from lynxius_evals.prompts.eval_prompt import EvalPromptTemplate


class CustomEval(Evaluator):
    def __init__(
        self,
        label: str,
        prompt_template: str,
        name: str = None,
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
        self.prompt_template = prompt_template
        self.name_override = name
        self.samples = []
        self.variables = [
            fname for _, fname, _, _ in Formatter().parse(prompt_template) if fname
        ]
        self.evaluated_results = None

    def add_trace(self, values: dict[str, str], context: list[ContextChunk] = []):

        # Ensure that all variables in the template are provided
        for var_name in self.variables:
            if var_name not in values:
                raise ValueError(f"Variable '{var_name}' is not provided.")

        # Ensure no extra variables were provided to reduce potential confusion
        variable_names_set = set(self.variables)
        for var_name, _ in values.items():
            if var_name not in variable_names_set:
                raise ValueError(
                    f"Variable '{var_name}' doesn't appear in the template."
                )

        self.samples.append((values, context))

    def get_url(self, run_local: bool = False):
        return "/evals/store/custom_eval/" if run_local else "/evals/run/custom_eval/"

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
        else:
            body = {
                "label": self.label,
                "href": self.href,
                "tags": self.tags,
                "baseline_project_uuid": self.baseline_project_uuid,
                "baseline_eval_run_label": self.baseline_eval_run_label,
                "prompt_template": self.prompt_template,
                "name_override": self.name_override,
                "data": [
                    {"variables": item[0], "contexts": [c.__dict__ for c in item[1]]}
                    for item in self.samples
                ],
            }

        return body

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
