from collections.abc import Mapping
from decimal import Decimal

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
            score = self.json_diff(o1, o2, weights)

            result.append(
                {
                    "reference": o1,
                    "output": o2,
                    "weights": weights,
                    "contexts": contexts,
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

        def sum_dict(d):
            total = 0
            if isinstance(d, list) or isinstance(d, str) or isinstance(d, bool):
                raise ValueError(
                    f"Weights object can contain only floats or ints, not: {d}"
                )
            elif isinstance(d, (int, float)):
                total += d
            elif isinstance(d, dict):
                for key, value in d.items():
                    parent_sum = sum_dict(value)
                    total += sum_dict(value)
                    if not (0.0 <= parent_sum <= 1.0):
                        raise ValueError(
                            f"The sum of the weights within key '{key}' is not within"
                            f" [0.0, 1.0], but is: {parent_sum}"
                        )
            return total

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

                weight = (
                    next_weights
                    if isinstance(next_weights, (int, float))
                    else sum_dict(next_weights)
                )

                base_weights.append(weight)
                base_scores.append(base_score)

            assert len(base_scores) == len(base_weights)

            # Cast to avoid floating-point precision error
            base_scores = [Decimal(s) for s in base_scores]
            base_weights = [Decimal(w) for w in base_weights]

            # Adjust the weights such that they sum to len(base_weights)
            factor = len(base_weights) / sum(base_weights)

            return float(
                sum(s * w * factor for (s, w) in zip(base_scores, base_weights))
                / Decimal(len(base_scores))
            )
        elif isinstance(o1, list) and isinstance(o2, list):
            if len(o1) == 0 and len(o2) == 0:
                return 1

            base_scores = [self.json_diff(e1, e2, weights) for (e1, e2) in zip(o1, o2)]
            base_scores = [s for s in base_scores if s is not None]

            # Cast to avoid floating-point precision error
            base_scores = [Decimal(s) for s in base_scores]

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
