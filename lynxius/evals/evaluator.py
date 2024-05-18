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

    def validate_tag(value):
        if "," in value or " " in value:
            raise ValueError(f"Tags can't contain spaces or commas: {value}")
