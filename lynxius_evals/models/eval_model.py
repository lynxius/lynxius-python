from abc import ABC, abstractmethod


class EvalModel(ABC):
    """An abstraction for a language model."""

    @abstractmethod
    def query(self, prompt: str) -> list[str]:
        """Create a chat completion.

        This method must be overridden in subclasses to provide the actual logic
        for creating an asynchronous chat completion.

        Args:
            prompt (str): The prompt to pass to the language model.

        Raises:
            NotImplementedError: When the method is not overridden in subclasses.

        Returns:
            list[str]: The model chat responses.
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, input: list[str]) -> list[list[float]]:
        """Create an embedding call.

        This method must be overridden in subclasses to provide the actual logic
        for creating an asynchronous embedding call.

        Args:
            input (list[str]): A list of sequences to embed.

        Raises:
            NotImplementedError: When the method is not overridden in subclasses.

        Returns:
            list[list[float]]: For every input sequence returns a list floating
            point numbers representing the embedding for that sequence.
        """
        raise NotImplementedError
