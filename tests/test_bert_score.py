import os
import numpy as np
import pytest
import yaml

from lynxius.client import LynxiusClient
from lynxius.evals.bert_score import BertScore

THRESHOLD = 0.1
scores = []
api_key = os.getenv("LYNXIUS_API_KEY")


@pytest.fixture(scope="class")
def yaml_data(request):
    relative_path = os.path.join(
        os.path.dirname(__file__),
        "test_bert_score.yaml",
    )
    TEST_FILE_PATH = os.path.normpath(relative_path)

    with open(TEST_FILE_PATH, "r") as file:
        data = yaml.safe_load(file)

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
class TestBertScore:
    """Test `BertScore` evaluator."""

    @pytest.fixture(scope="class")
    def init_client(self):
        return LynxiusClient(api_key=api_key)

    @pytest.fixture(scope="class")
    def run_eval(self, init_client):
        label = "unit_test"
        tags = ["bert_score"]
        eval = BertScore(label=label, tags=tags, level="word", presence_threshold=0.55)

        for entry in self.yaml_data["test_cases"]:
            eval.add_trace(
                reference=entry["input"]["reference"],
                output=entry["input"]["output"],
            )

        bert_score_uuid = init_client.evaluate(eval)
        assert (
            bert_score_uuid is not None
        ), "Failed to get a valid UUID from evaluate method"

        eval_run = init_client.get_eval_run(bert_score_uuid)
        assert eval_run is not None, "Failed to retrieve evaluation run"

        if "results" in eval_run and eval_run["results"]:
            for entry in eval_run["results"]:
                f1 = entry.get("f1")
                if f1 is not None:
                    scores.append(float(f1))

        return eval_run

    def test_init(self, run_eval):
        assert run_eval["label"] == "unit_test"
        assert "bert_score" in run_eval["tags"]

    def test_statistics(self, run_eval):
        assert run_eval is not None, "Evaluation run should not be None"
        expected_average_score, expected_percentile_20, expected_percentile_90 = (
            calculate_statistics(scores)
        )

        if expected_average_score is None:
            pytest.fail("No valid scores to calculate statistics.")

        aggregate_score = float(run_eval.get("aggregate_score"))
        p20 = float(run_eval.get("p20"))
        p90 = float(run_eval.get("p90"))

        assert abs(expected_average_score - aggregate_score) <= THRESHOLD
        assert abs(expected_percentile_20 - p20) <= THRESHOLD
        assert abs(expected_percentile_90 - p90) <= THRESHOLD

    def test_evaluate_high_score(self, run_eval):
        assert run_eval is not None, "Evaluation run should not be None"
        entry = self.yaml_data["test_cases"][0]
        expected_score = float(entry["expected_output"])

        f1 = run_eval["results"][0].get("f1")
        assert f1 is not None, "Score should not be None"
        assert abs(float(f1) - expected_score) <= THRESHOLD

    def test_evaluate_medium_score(self, run_eval):
        assert run_eval is not None, "Evaluation run should not be None"
        entry = self.yaml_data["test_cases"][1]
        expected_score = float(entry["expected_output"])

        f1 = run_eval["results"][1].get("f1")
        assert f1 is not None, "Score should not be None"
        assert abs(float(f1) - expected_score) <= THRESHOLD

    def test_evaluate_low_score(self, run_eval):
        assert run_eval is not None, "Evaluation run should not be None"
        entry = self.yaml_data["test_cases"][2]
        expected_score = float(entry["expected_output"])

        f1 = run_eval["results"][2].get("f1")
        assert f1 is not None, "Score should not be None"
        assert abs(float(f1) - expected_score) <= THRESHOLD
