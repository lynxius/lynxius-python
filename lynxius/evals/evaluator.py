from lynxius.tracing.observe import _try_register_eval
from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    A base class for all Lynxius evals
    """

    @abstractmethod
    def get_url(self, run_local: bool = False):
        return NotImplemented

    @abstractmethod
    def get_request_body(self, run_local: bool = False):
        return NotImplemented

    @abstractmethod
    def get_merge_id(self) -> int:
        """
        This function has to return a stable hash of values that uniquely identify an
        eval run. This function will be used to merge the results of multiple evaluators
        of the same type into a single eval run.
        Generally speaking, computing the hash of all constructor parameter values as
        well as the type of the evaluator should do the job for this.
        """
        return NotImplemented

    def evaluate_local(self):
        # If there's an ongoing trace, assign this evaluator to that trace
        _try_register_eval(self)

    def validate_tag(value):
        if "," in value or " " in value:
            raise ValueError(f"Tags can't contain spaces or commas: {value}")
