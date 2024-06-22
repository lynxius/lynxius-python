from lynxius.evals.bert_score import BertScore
from lynxius_evals.evaluators.bert_score_eval import BertScoreEval
from lynxius.evals.local.evaluator_local import EvaluatorLocal
from lynxius_evals.models.openai import OpenAIModel


class BertScoreLocal(BertScore, EvaluatorLocal):
    def __init__(
        self,
        label: str,
        level: str = "word",
        presence_threshold: float = 0.65,
        href: str = None,
        tags: list[str] = [],
    ):
        super().__init__(label, level, presence_threshold, href, tags)

        self.evaluated_results = None

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

    def get_url(self):
        return "/evals/store/bert_score/"

    def get_request_body(self):
        if self.evaluated_results is None:
            raise Exception(
                "Call evaluator.evaluate() first and then store the evaluator output"
            )

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

        return body
