import os
import numpy as np
import pytest
import yaml

from lynxius.client import LynxiusClient
from lynxius.evals.context_precision import ContextPrecision
from lynxius.rag.types import ContextChunk

THRESHOLD = 0.1
scores = []
api_key = os.getenv("LYNXIUS_API_KEY")

@pytest.fixture(scope="class")
def yaml_data(request):
    relative_path = os.path.join(
        os.path.dirname(__file__),
        "test_context_precision.yaml",
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
class TestContextPrecision:
    """Test `ContextPrecision` evaluator."""

    @pytest.fixture(scope="class")
    def init_client(self):
        return LynxiusClient(api_key=api_key)

    @pytest.fixture(scope="class")
    def run_eval(self, init_client):
        label = "unit_test"
        tags = ["context_precision"]
        eval = ContextPrecision(label=label, tags=tags)

        for entry in self.yaml_data['test_cases']:
            contexts = [ContextChunk(document=doc["document"], relevance=doc["relevance"]) for doc in entry["input"]["contexts"]]
            eval.add_trace(
                query=entry["input"]["query"],
                reference=entry["input"]["reference"],
                context=contexts
            )

        try:
            context_precision_uuid = init_client.evaluate(eval)
            assert context_precision_uuid is not None, "Failed to get a valid UUID from evaluate method"

            eval_run = init_client.get_eval_run(context_precision_uuid)
            assert eval_run is not None, "Failed to retrieve evaluation run"

            if "results" in eval_run and eval_run["results"]:
                for entry in eval_run["results"]:
                    score = entry.get("score")
                    if score is not None:
                        scores.append(float(score))

            return eval_run
        except Exception as e:
            pytest.fail(f"Evaluation failed: {str(e)}")

    def test_init(self, run_eval):
        assert run_eval["label"] == "unit_test"
        assert "context_precision" in run_eval["tags"]

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
        entry = self.yaml_data['test_cases'][0]
        expected_score = float(entry["expected_output"])

        score = run_eval["results"][0].get("score")
        assert score is not None, "Score should not be None"
        assert abs(float(score) - expected_score) <= THRESHOLD

    def test_evaluate_low_score(self, run_eval):
        assert run_eval is not None, "Evaluation run should not be None"
        entry = self.yaml_data['test_cases'][1]
        expected_score = float(entry["expected_output"])

        score = run_eval["results"][1].get("score")
        assert score is not None, "Score should not be None"
        assert abs(float(score) - expected_score) <= THRESHOLD

    def test_evaluate_medium_score(self, run_eval):
        assert run_eval is not None, "Evaluation run should not be None"
        entry = self.yaml_data['test_cases'][2]
        expected_score = float(entry["expected_output"])

        score = run_eval["results"][2].get("score")
        assert score is not None, "Score should not be None"
        assert abs(float(score) - expected_score) <= THRESHOLD
