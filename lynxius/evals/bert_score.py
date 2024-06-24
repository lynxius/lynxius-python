from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk
from lynxius_evals.evaluators.bert_score_eval import BertScoreEval
from lynxius_evals.models.openai import OpenAIModel


class BertScore(Evaluator):
    def __init__(
        self,
        label: str,
        level: str = "word",
        presence_threshold: float = 0.65,
        href: str = None,
        tags: list[str] = [],
    ):
        levels = ["word", "sentence"]
        if level not in levels:
            raise ValueError(f"Level must be one of the following: {levels}")

        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.level = level
        self.presence_threshold = presence_threshold
        self.samples = []
        self.evaluated_results = None

    def add_trace(self, reference: str, output: str, context: list[ContextChunk] = []):
        if not reference or not output:
            raise ValueError("Both reference and output must be provided")

        self.samples.append((reference, output, context))

    def get_url(self, run_local: bool = False):
        return "/evals/store/bert_score/" if run_local else "/evals/run/bert_score/"

    def get_request_body(self, run_local: bool = False):
        if run_local and self.evaluated_results is None:
            raise Exception(
                "Call evaluate_local() before storing the local evaluation output"
            )

        if run_local:
            body = {
                "label": self.label,
                "href": self.href,
                "tags": self.tags,
                "level": self.level,
                "presence_threshold": self.presence_threshold,
                "version": "1",
                "data": [
                    {
                        "reference": result["reference"],
                        "output": result["output"],
                        "precision": result["precision"],
                        "recall": result["recall"],
                        "f1": result["f1"],
                        "missing_tokens": result["missing_tokens"],
                        "contexts": [c.__dict__ for c in result["contexts"]],
                    }
                    for result in self.evaluated_results
                ],
            }
        else:
            body = {
                "label": self.label,
                "href": self.href,
                "tags": self.tags,
                "level": self.level,
                "presence_threshold": self.presence_threshold,
                "data": [
                    {
                        "reference": item[0],
                        "output": item[1],
                        "contexts": [c.__dict__ for c in item[2]],
                    }
                    for item in self.samples
                ],
            }

        return body

    def evaluate_local(self):
        model = OpenAIModel(embedding_model="text-embedding-3-small")
        eval = BertScoreEval(model, self.level, self.presence_threshold)

        variables = []
        for sample in self.samples:
            variables.append(
                {
                    "reference": sample[0],
                    "output": sample[1],
                    "contexts": sample[2],
                }
            )

        self.evaluated_results = eval.evaluate(variables)
