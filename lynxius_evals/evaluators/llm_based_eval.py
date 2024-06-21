import concurrent.futures
import logging
from collections.abc import Mapping
from typing import Any

import tiktoken
from lynxius_evals.models.eval_model import EvalModel
from lynxius_evals.models.openai import (
    DEFAULT_MAX_TOKENS_GPT_4,
    DEFAULT_OPENAI_MODEL,
)
from lynxius_evals.prompts.eval_prompt import EvalPromptTemplate

logger = logging.getLogger(__name__)


class TokenLimitExceededError(Exception):
    """Exception raised when the token count exceeds the maximum allowed limit."""

    def __init__(self, message="Token count exceeds the maximum allowed limit"):
        self.message = message
        super().__init__(self.message)


class LLMBasedEval:
    def __init__(
        self,
        model: EvalModel,
        template: EvalPromptTemplate,
        output_map: Mapping[str, Any] | None = None,
        output_default: Any | None = None,
        max_workers: int | None = None,
    ):
        """
        Args:
            model: LLM model `EvalModel`.

            template: LLM template `EvalPromptTemplate`.

            output_map: defines how the output of an LLM will be converted to the score.
                `{"correct": True, "incorrect": False}` will convert the textual LLM
                output to a boolean value.
                The LLM output will first be converted to lower case characters and all
                leading/trailing whitespace will be removed.

            output_default: if the LLM output didn't match any of the keys provided in
                `output_map`, `output_default` will be used for a score.

            max_workers: How many threads to use. Gets passed down to
                `ThreadPoolExecutor`.
        """
        if not model:
            raise ValueError("Model has to be provided")
        if not template:
            raise ValueError("Template has to be provided")
        if output_map is None or output_default is None:
            if type(self).parse_output == LLMBasedEval.parse_output:
                raise ValueError(
                    "You either have to override parse_output or provide output_map"
                    "and output_default."
                )

        self.model = model
        self.template = template
        self.output_map = output_map
        self.output_default = output_default
        self.max_workers = max_workers

    def evaluate(
        self,
        variable_values_list: list[Mapping[str, str | list[str]]],
    ) -> list[Mapping]:
        result = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit background tasks
            future_to_index = {}
            for i, variable_values in enumerate(variable_values_list):
                formatted_template = self.format_template(**variable_values)

                # Tokenize and count
                if self.model == DEFAULT_OPENAI_MODEL:
                    enc = tiktoken.encoding_for_model(DEFAULT_OPENAI_MODEL)
                    tokens = enc.encode(formatted_template)
                    token_count = len(tokens)
                    if token_count > DEFAULT_MAX_TOKENS_GPT_4:
                        # TODO: this will crash an eval that we've already charged our
                        # customer for. We should somehow fix this. Also, our customer
                        # will not know what's the problem - the eval will simply not
                        # show up in the EvalRuns page.
                        raise TokenLimitExceededError(
                            f"Token count ({token_count}) exceeds the maximum allowed \
                                for {self.model}."
                        )

                future = executor.submit(self.model.query, formatted_template)
                future_to_index[future] = i

                # Append the input values to the result.
                # Outputs will be appended as they come.
                result.append({**variable_values, "llm_input": formatted_template})

            # Wait for tasks to complete
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    llm_output = future.result()[0]
                    score = self.parse_output(llm_output, variable_values_list[index])
                    result[index].update(
                        {
                            "llm_output": llm_output,
                            "score": score,
                        }
                    )
                except Exception as e:
                    logger.exception(e)

        return result

    def format_template(self, **variable_values: Mapping[str, str | list[str]]):
        return self.template.format(**variable_values)

    def parse_output(self, output: str, variable_values: Mapping[str, str | list[str]]):
        """
        Extracts the score from the LLM output.
        This function assumes that the score is on the very last non-empty line of
        the output. The LLM can produce more information like a preamble or an
        explanation of its verdict. All this information has to be above and the final
        verdict which is always on a last non-empty line.
        """

        score = output.lower().strip().splitlines()[-1]
        for key, value in self.output_map.items():
            if score == key.lower():
                return value
        return self.output_default
