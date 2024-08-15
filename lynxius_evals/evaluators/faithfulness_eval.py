from collections.abc import Mapping
import json
import logging
from json.decoder import JSONDecodeError
from lynxius_evals.evaluators.llm_based_eval import LLMBasedEval

logger = logging.getLogger(__name__)

class FaithfulnessEval(LLMBasedEval):
    """
    A `FaithfulnessEval` evaluator class for assessing the factual consistency (faithfulness) of a
    candidate answer given a reference text. The score is in the range of [0.0, 1.0].

    This evaluator classifies the statements in the answer into two categories:

    - "faithful": statements in the answer that CAN be directly inferred from the reference text
    - "unfaithful": statements in the answer that CANNOT be directly inferred from the reference text

    And then computes the score using the formula:
    faithfulness = |number of faithful statements| / |total number of statements|
    """

    def parse_output(
        self, output: str, variable_values: Mapping[str, str | list[str]]
    ) -> float:
        try:
            result = json.loads(output)
            faithful = result["faithful"]
            unfaithful = result["unfaithful"]
        except JSONDecodeError as e:
            logger.exception(e)
            logger.error(output)
            return 0.0

        if not isinstance(faithful, list) or not isinstance(unfaithful, list):
            logger.error("Evaluator didn't return valid JSON lists")
            logger.error(output)
            return 0.0

        # Calculate the faithfulness score
        total_statements = len(faithful) + len(unfaithful)
        if total_statements == 0:
            return 0.0

        score = len(faithful) / total_statements
        return score
