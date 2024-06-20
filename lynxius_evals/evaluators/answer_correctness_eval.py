from collections.abc import Mapping
import json
import logging
from json.decoder import JSONDecodeError
from lynxius_evals.evaluators.llm_based_eval import LLMBasedEval


logger = logging.getLogger(__name__)


class AnswerCorrectnessEval(LLMBasedEval):
    """An `AnswerCorrectnessEval` evaluator class for assessing correctness of a
    candidate answer given a reference answer. The score is in the range of [0.0, 1.0].

    This evaluator classifies the provided statements into 3 categories:

    - TP (true positive): statements that are present in answer that are also directly
      supported by the one or more statements in reference,
    - FP (false positive): statements present in the answer but not directly supported
      by any statement in reference,
    - FN (false negative): statements found in the reference but not present in answer.

    And then computes the score like so:
    f1 = |TP| / (|TP| + 0.5 * (|FP| + |FN|))
    """

    def parse_output(
        self, output: str, variable_values: Mapping[str, str | list[str]]
    ) -> float:
        try:
            result = json.loads(output)
            tp = result["TP"]
            fp = result["FP"]
            fn = result["FN"]
        except JSONDecodeError as e:
            logger.exception(e)
            logger.error(output)
            return 0.0

        if (
            not isinstance(tp, list)
            or not isinstance(fp, list)
            or not isinstance(fn, list)
        ):
            logger.error("Evaluator didn't return valid JSON lists")
            logger.error(output)
            return 0.0

        # TODO: If we error out in one of the checks above, it's probably our fault.
        # We might want to consider refunding our customers in such case.

        # All seems good, let's compute the score.
        # The following code is heavily inspired by Ragas.

        f1 = len(tp) / (len(tp) + 0.5 * (len(fp) + len(fn)))
        return f1
