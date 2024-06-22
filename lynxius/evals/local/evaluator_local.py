from lynxius.evals.evaluator import Evaluator
from abc import abstractmethod


class EvaluatorLocal(Evaluator):
    """
    A base class for all Lynxius evals that are executed locally
    """

    @abstractmethod
    def evaluate_local(self):
        return NotImplemented
