from collections.abc import Mapping

from Levenshtein import distance


class JsonDiffEval:
    """A `JsonDiffEval` evaluator class for assessing the similarity of two JSON
    objects using normalized difference for numeric types (int, float and bool) and
    Levenshtein distance for strings.
    """

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        variable_values: list[Mapping[str, str]],
    ) -> list[dict]:
        result = []
        for item in variable_values:
            o1 = item["reference"]
            o2 = item["output"]
            weights = item.get("weights", {})
            contexts = item["contexts"]
            trace_uuid = item["trace_uuid"]
            score = self.json_diff(o1, o2, weights)

            result.append(
                {
                    "reference": o1,
                    "output": o2,
                    "weights": weights,
                    "contexts": contexts,
                    "trace_uuid": trace_uuid,
                    "score": score,
                }
            )

        return result

    def json_diff(self, o1, o2, weights={}):
        def get_next_weights(prev_weights, key):
            if isinstance(prev_weights, (int, float)):
                return 1.0
            else:
                return weights.get(key, 1.0)

        def get_levenshtein_distance(o1: str, o2: str):
            max_len = max(len(x) for x in [o1, o2])

            # Normalized Levenshtein distance in [0,1]
            score = 1
            if max_len > 0:
                score = 1 - (distance(o1, o2) / max_len)

            return score

        if isinstance(o1, dict) and isinstance(o2, dict):
            if len(o1) == 0 and len(o2) == 0:
                return 1

            # We only look at the intersection of the keys
            # Mismatch is not penalized in any way
            all_keys = set(o1.keys()).union(set(o2.keys()))

            base_scores = []
            base_weights = []
            for key in all_keys:
                next_weights = get_next_weights(weights, key)
                base_score = self.json_diff(o1.get(key), o2.get(key), next_weights)
                if base_score is None:
                    continue

                loopback_key = "__" + key
                if isinstance(next_weights, (int, float)):
                    weight = next_weights
                elif isinstance(next_weights, dict) and loopback_key in next_weights:
                    weight = next_weights[loopback_key]
                else:
                    weight = 1.0

                base_weights.append(weight)
                base_scores.append(base_score)

            assert len(base_scores) == len(base_weights)

            # Adjust the weights such that they sum to len(base_weights)
            if sum(base_weights) == 0:
                # Weights can't sum up to 0. Assume weights for all keys are 1.
                base_weights = [1 for _ in base_weights]
                factor = 1.0
            else:
                factor = len(base_weights) / sum(base_weights)

            return sum(
                s * w * factor for (s, w) in zip(base_scores, base_weights)
            ) / len(base_scores)
        elif isinstance(o1, list) and isinstance(o2, list):
            if len(o1) == 0 and len(o2) == 0:
                return 1

            base_scores = [self.json_diff(e1, e2, weights) for (e1, e2) in zip(o1, o2)]
            base_scores = [s for s in base_scores if s is not None]
            return sum(base_scores) / len(base_scores)
        elif isinstance(o1, str) and isinstance(o2, str):
            return get_levenshtein_distance(o1, o2)
        elif (isinstance(o1, int) or isinstance(o1, float)) and (
            isinstance(o2, int) or isinstance(o2, float)
        ):
            if o2 == 0 and o1 == 0:
                return 1
            else:
                # Normalized difference
                return 1 - abs(o2 - o1) / (abs(o2) + abs(o1))
        elif o1 is None and o2 is None:
            return 1
        elif o1 is None or o2 is None:
            return 0
        else:
            return 0
