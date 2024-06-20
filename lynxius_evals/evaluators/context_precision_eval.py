from collections.abc import Mapping
import json
import logging
from json.decoder import JSONDecodeError
import numpy as np
from lynxius_evals.evaluators.llm_based_eval import LLMBasedEval


logger = logging.getLogger(__name__)


class ContextPrecisionEval(LLMBasedEval):
    """A `ContextPrecisionEval` evaluator class for assessing the quality of the
    context in a RAG application. The score is in the range of [0.0, 1.0].

    This evaluator penalizes relevant context chunks stored in low ranks. This means
    that in order for this metric to have a high (good) value, all relevant context
    chunks should be higher than the non-relevant ones.
    """

    def format_template(self, **variable_values: Mapping[str, str | list[str]]):
        # Replace contexts list to a single context variable
        variable_values_copy = {
            k: v for k, v in variable_values.items() if k != "contexts"
        }

        context = ""
        for c in variable_values["contexts"]:
            document = c["document"]
            context += f"************\n[Context]: {document}\n"

        variable_values_copy["context"] = context

        return self.template.format(**variable_values_copy)

    def parse_output(
        self, output: str, variable_values: Mapping[str, str | list[str]]
    ) -> float:
        try:
            verdicts = json.loads(output)
            verdicts = verdicts["result"]
        except JSONDecodeError as e:
            logger.exception(e)
            logger.error(output)
            return 0.0

        if not isinstance(verdicts, list):
            logger.error("Evaluator didn't return a valid JSON list")
            logger.error(output)
            return 0.0

        num_contexts = len(variable_values["contexts"])
        if len(verdicts) != num_contexts:
            v = len(verdicts)
            c = num_contexts
            logger.error(
                f"Evaluator returned {v} verdicts but there were {c} context chunks"
            )
            logger.error(output)
            return 0.0

        # TODO: If we error out in one of the checks above, it's probably our fault.
        # We might want to consider refunding our customers in such case.

        # All seems good, let's compute the score.
        # The following code is heavily inspired by Ragas.
        score = np.nan

        verdict_list = [1 if ver else 0 for ver in verdicts]
        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )

        score = numerator / denominator
        if np.isnan(score):
            logger.warning("Invalid response format - computation ended up with a NaN")

        return score
