from string import Formatter
from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk


class CustomEval(Evaluator):
    def __init__(
        self,
        label: str,
        prompt_template: str,
        name: str = None,
        href: str = None,
        tags: list[str] = [],
    ):
        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.prompt_template = prompt_template
        self.name_override = name
        self.samples = []
        self.variables = [
            fname for _, fname, _, _ in Formatter().parse(prompt_template) if fname
        ]

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

    def get_url(self):
        return "/evals/run/custom_eval/"

    def get_request_body(self):
        body = {
            "label": self.label,
            "href": self.href,
            "tags": self.tags,
            "prompt_template": self.prompt_template,
            "name_override": self.name_override,
            "data": [
                {"variables": item[0], "contexts": [c.__dict__ for c in item[1]]}
                for item in self.samples
            ],
        }

        return body
