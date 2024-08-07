import os
import pytest
import yaml
import numpy as np
from lynxius.client import LynxiusClient
from lynxius.evals.context_precision import ContextPrecision
from lynxius.rag.types import ContextChunk

THRESHOLD = 0.15
api_key = os.getenv('LYNXIUS_API_KEY')


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


@pytest.fixture(scope="class")
def init_client():
    return LynxiusClient(api_key=api_key)


@pytest.fixture(scope="class")
def init_model():
    label = "iter_1"
    tags = ["context_precision"]
    return ContextPrecision(label=label, tags=tags)


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
    def run_eval(self, init_client, init_model):
        scores = []
        for entry in self.yaml_data['test_cases']:
            init_model.add_trace(
                query=entry["input"]["query"],
                reference=entry["input"]["reference"],
                context=[ContextChunk(document=ctx["document"], relevance=ctx["relevance"]) for ctx in entry["input"]["contexts"]]
            )

        eval_run_id = init_client.evaluate(init_model)
        eval_run = init_client.get_eval_run(eval_run_id)
        scores.extend([float(result.get("score")) for result in eval_run["results"]])
        
        eval_run["statistics"] = calculate_statistics(scores)
        return eval_run

    def test_init(self, init_model):
        assert init_model.label == "iter_1"
        assert "context_precision" in init_model.tags

    def test_yaml_data(self):
        # Access the YAML data attached to the class
        assert len(self.yaml_data['test_cases']) == 18

    def test_evaluate_high_score(self, run_eval):
        entry = self.yaml_data['test_cases'][0]
        expected_score = float(entry["expected_output"])

        result_score = float(run_eval["results"][0].get("score"))
        assert abs(result_score - expected_score) <= THRESHOLD

    def test_evaluate_low_score(self, run_eval):
        entry = self.yaml_data['test_cases'][1]
        expected_score = float(entry["expected_output"])

        result_score = float(run_eval["results"][1].get("score"))
        assert abs(result_score - expected_score) <= THRESHOLD

    def test_evaluate_medium_score(self, run_eval):
        entry = self.yaml_data['test_cases'][2]
        expected_score = float(entry["expected_output"])

        result_score = float(run_eval["results"][2].get("score"))
        assert abs(result_score - expected_score) <= THRESHOLD

    def test_statistics(self, run_eval):
        expected_average_score, expected_percentile_20, expected_percentile_90 = run_eval["statistics"]

        aggregate_score = float(run_eval.get("aggregate_score"))
        p20 = float(run_eval.get("p20"))
        p90 = float(run_eval.get("p90"))

        assert abs(expected_average_score - aggregate_score) <= THRESHOLD
        assert abs(expected_percentile_20 - p20) <= THRESHOLD
        assert abs(expected_percentile_90 - p90) <= THRESHOLD