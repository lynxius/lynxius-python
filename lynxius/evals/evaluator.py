from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    A base class for all Lynxius evals
    """

    @abstractmethod
    def get_url(self):
        return NotImplemented

    @abstractmethod
    def get_request_body(self):
        return NotImplemented
