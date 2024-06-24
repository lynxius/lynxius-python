from lynxius.evals.evaluator import Evaluator
from lynxius.rag.types import ContextChunk
from lynxius_evals.evaluators.semantic_similarity_eval import SemanticSimilarityEval
from lynxius_evals.models.openai import OpenAIModel


class SemanticSimilarity(Evaluator):
    def __init__(
        self,
        label: str,
        href: str = None,
        tags: list[str] = [],
    ):
        [Evaluator.validate_tag(value) for value in tags]

        self.label = label
        self.href = href
        self.tags = tags
        self.samples = []
        self.evaluated_results = None

    def add_trace(self, reference: str, output: str, context: list[ContextChunk] = []):
        if not reference or not output:
            raise ValueError("Both reference and output must be provided")

        self.samples.append((reference, output, context))

    def get_url(self, run_local: bool = False):
        return (
            "/evals/store/semantic_similarity/"
            if run_local
            else "/evals/run/semantic_similarity/"
        )

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
                "version": "1",
                "data": [
                    {
                        "reference": result["reference"],
                        "output": result["output"],
                        "similarity": result["similarity"],
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
        model = OpenAIModel()
        eval = SemanticSimilarityEval(model)

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
