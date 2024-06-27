from openai import OpenAI

from .eval_model import EvalModel

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MAX_TOKENS_GPT_4 = 4096
DEFAULT_RESPONSE_FORMAT = "text"


class OpenAIModel(EvalModel):
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
    ):
        """
        response_format can either be `default` or `json`. This value will be passed
        down to the OpenAI client.
        """

        if response_format not in ["text", "json_object"]:
            return ValueError(
                "response_format either can be 'text' or 'json_object',"
                f" not '{response_format}'"
            )

        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.response_format = response_format
        self.client = OpenAI()

    def query(self, prompt: str) -> list[str]:

        # Preprocess messages
        messages_processed = []

        # TODO: we might want to experiment with system messages:
        # messages_processed.append(
        #     {"role": "system", "content": "You are a responsible LLM evaluator ..."}
        # )

        if isinstance(prompt, str):
            messages_processed.append({"role": "user", "content": prompt})
        else:
            messages_processed.append(prompt)

        completion = self.client.chat.completions.create(
            messages=messages_processed,
            model=self.model,
            temperature=self.temperature,
            response_format={"type": self.response_format},
        )

        result = []
        for choice in completion.choices:
            result.append(choice.message.content)

        return result

    def embed(self, input: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=input,
            model=self.embedding_model,
        )

        response = [data.embedding for data in response.data]
        return response
