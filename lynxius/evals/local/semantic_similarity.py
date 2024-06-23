from lynxius.evals.semantic_similarity import SemanticSimilarity
from lynxius_evals.evaluators.semantic_similarity_eval import SemanticSimilarityEval
from lynxius.evals.local.evaluator_local import EvaluatorLocal
from lynxius_evals.models.openai import OpenAIModel


class SemanticSimilarityLocal(SemanticSimilarity, EvaluatorLocal):
    def __init__(self, label: str, href: str = None, tags: list[str] = []):
        super().__init__(label, href, tags)

        self.evaluated_results = None

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

    def get_url(self):
        return "/evals/store/semantic_similarity/"

    def get_request_body(self):
        if self.evaluated_results is None:
            raise Exception(
                "Call evaluator.evaluate_local() first and then store the evaluator output"
            )

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

        return body
