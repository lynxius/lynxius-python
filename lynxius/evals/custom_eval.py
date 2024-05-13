from string import Formatter
from lynxius.evals.evaluator import Evaluator


class CustomEval(Evaluator):
    def __init__(self, title: str, prompt_template: str):
        self.title = title
        self.prompt_template = prompt_template
        self.samples = []
        self.variables = [
            fname for _, fname, _, _ in Formatter().parse(prompt_template) if fname
        ]

    def add_trace(self, values: dict[str, str]):

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

        values["contexts"]=[]
        self.samples.append(values)

    def get_url(self):
        return "/api/evals/run/custom_eval/"

    def get_request_body(self):
        body = {
            "title": self.title,
            "prompt_template": self.prompt_template,
            "data": [item for item in self.samples],
        }

        return body
