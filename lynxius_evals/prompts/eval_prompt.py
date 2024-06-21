from string import Formatter


class EvalPromptTemplate:
    def __init__(self, name: str, template: str):
        """
        Args:
            name (str): The prompt name.
            template (str): The prompt template with variables in curly braces {}.
                If you need to use curly braces without having them formatted, you have
                to escape them with double curly braces.
                Will be formatted: {variable}
                Will not be formatted: {{variable}}
        """
        self.name = name
        self.template = template

    def get_variables(self) -> list[str]:
        variables = [
            fname for _, fname, _, _ in Formatter().parse(self.template) if fname
        ]
        return variables

    def format(self, **kwargs) -> str:
        """
        Please provide the variable values as key-value arguments to this function.
        """
        # Ensure that all variables in the template are provided
        variable_names = self.get_variables()
        for var_name in variable_names:
            if var_name not in kwargs:
                raise ValueError(f"Variable '{var_name}' is not provided.")

        return self.template.format(**kwargs)
