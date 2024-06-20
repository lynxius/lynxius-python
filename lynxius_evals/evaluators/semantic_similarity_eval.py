from collections.abc import Mapping

import numpy as np

from lynxius_evals.models.eval_model import EvalModel


class SemanticSimilarityEval:
    """A `SemanticSimilarity` evaluator class for assessing semantic similarity
    between the answer produced by an LLM and the ground truth answer. The score
    is in the range of [0.0, 1.0].
    """

    def __init__(self, model: EvalModel) -> None:
        self.model = model

    def evaluate(
        self,
        variable_values_list: list[Mapping[str, str]],
    ) -> list[float]:

        texts_to_embed = []
        for variable_values in variable_values_list:
            texts_to_embed.append(variable_values["reference"])
            texts_to_embed.append(variable_values["output"])

        embeddings = self.model.embed(texts_to_embed)

        assert len(embeddings) == len(variable_values_list) * 2

        result = []
        for i in range(0, len(embeddings), 2):
            reference = variable_values_list[i // 2]["reference"]
            output = variable_values_list[i // 2]["output"]
            contexts = variable_values_list[i // 2]["contexts"]
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i + 1])

            # According to OpenAI docs, their embeddings are already normalized to have
            # a norm of 1, so cosine similarity (as well as Euclidian distance) is
            # effectively a dot product:
            # https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use
            # We divide by the product of the two norms anyway to make a potential
            # future transition to a non-normalized embeddings provider easier.
            cosine_similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )

            # In theory, cosine similarity is in the range of [-1, 1]. In practice
            # however, embeddings produced by LLMs tend to never have a negative cosine
            # similarity. This bias is most probably an artefact of various training
            # techniques like Max Pooling used while training. In case we get a
            # negative value, we simply clamp it to 0.
            cosine_similarity = np.clip(cosine_similarity, 0.0, 1.0)

            result.append(
                {
                    "reference": reference,
                    "output": output,
                    "contexts": contexts,
                    "similarity": float(cosine_similarity),
                }
            )

        return result
