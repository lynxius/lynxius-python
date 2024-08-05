import os

import numpy as np
import pytest
import yaml

from lynxius.client import LynxiusClient
from lynxius.evals.answer_correctness import AnswerCorrectness

THRESHOLD = 0.1
scores = []
api_key = os.getenv("LYNXIUS_API_KEY")


@pytest.fixture(scope="class")
def yaml_data(request):
    relative_path = os.path.join(
        os.path.dirname(__file__),
        "answer_correctness_data.yaml",
    )
    TEST_FILE_PATH = os.path.normpath(relative_path)

    # Open the YAML file
    with open(TEST_FILE_PATH, "r") as file:
        data = yaml.safe_load(file)

    # Attach the data to the class under test
    if request.cls is not None:
        request.cls.yaml_data = data

    yield


def calculate_statistics(scores):
    # Calculate average
    average_score = np.mean(scores)

    # Calculate 20th and 90th percentiles
    percentile_20 = np.percentile(scores, 20, method="inverted_cdf")
    percentile_90 = np.percentile(scores, 90, method="inverted_cdf")

    return average_score, percentile_20, percentile_90


@pytest.mark.usefixtures("yaml_data")
class TestAnswerCorrectness:
    """Test `AnswerCorrectness` evaluator."""

    @pytest.fixture(scope="class")
    def init_client(self):
        return LynxiusClient(api_key=api_key)

    @pytest.fixture(scope="class")
    def run_eval(self, init_client):
        label = "unit_test"
        tags = ["answer_correctness"]
        eval = AnswerCorrectness(label=label, tags=tags)

        for entry in self.yaml_data:
            eval.add_trace(
                query=entry["query"],
                reference=entry["reference"],
                output=entry["output"],
            )

        answer_correctness_uuid = init_client.evaluate(eval)
        eval_run = init_client.get_eval_run(answer_correctness_uuid)

        scores.extend([float(entry.get("score")) for entry in eval_run["results"]])

        return eval_run

    def test_init(self, run_eval):
        assert run_eval["label"] == "unit_test"
        assert "answer_correctness" in run_eval["tags"]

    def test_statistics(self, run_eval):
        expected_average_score, expected_percentile_20, expected_percentile_90 = (
            calculate_statistics(scores)
        )

        aggregate_score = float(run_eval.get("aggregate_score"))
        p20 = float(run_eval.get("p20"))
        p90 = float(run_eval.get("p90"))

        print(expected_average_score, expected_percentile_20, expected_percentile_90)
        print(aggregate_score, p20, p90)

        assert abs(expected_average_score - aggregate_score) <= THRESHOLD
        assert abs(expected_percentile_20 - p20) <= THRESHOLD
        assert abs(expected_percentile_90 - p90) <= THRESHOLD

    def test_evaluate_success(self, run_eval):
        entry = self.yaml_data[0]
        expected_score = float(entry["score"])

        score = float(run_eval["results"][0].get("score"))

        assert abs(score - expected_score) <= THRESHOLD

    def test_evaluate_failure(self, run_eval):
        entry = self.yaml_data[1]
        expected_score = float(entry["score"])

        score = float(run_eval["results"][1].get("score"))

        assert abs(score - expected_score) <= THRESHOLD

    def test_evaluate_half_right(self, run_eval):
        entry = self.yaml_data[2]
        expected_score = float(entry["score"])

        score = float(run_eval["results"][2].get("score"))

        assert abs(score - expected_score) <= THRESHOLD
