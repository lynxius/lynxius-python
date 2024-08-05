import os

import numpy as np
import pytest
import yaml

from lynxius.evals.answer_correctness import AnswerCorrectness

scores = []
THRESHOLD = 0.1


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


def calculate_percentiles(scores, percentile):
    return np.percentile(scores, percentile)


@pytest.mark.usefixtures("yaml_data")
class TestAnswerCorrectness:
    """Test `AnswerCorrectness` evaluator."""

    @pytest.fixture
    def run_eval(self):
        label = "unit_test"
        tags = ["answer_correctness"]
        eval = AnswerCorrectness(label=label, tags=tags)

        for entry in self.yaml_data:
            eval.add_trace(
                query=entry["query"],
                reference=entry["reference"],
                output=entry["output"],
            )

        eval.evaluate_local()
        json = eval.get_request_body(run_local=True)
        return json

    def test_init(self, run_eval):
        assert run_eval["label"] == "unit_test"
        assert "answer_correctness" in run_eval["tags"]

    def test_evaluate_success(self, run_eval):
        entry = self.yaml_data[0]
        expected_score = float(entry["score"])

        score = float(run_eval["data"][0].get("score"))
        scores.append(score)

        assert abs(score - expected_score) <= THRESHOLD

    def test_evaluate_failure(self, run_eval):
        entry = self.yaml_data[1]
        expected_score = float(entry["score"])

        score = float(run_eval["data"][1].get("score"))
        scores.append(score)

        assert abs(score - expected_score) <= THRESHOLD

    def test_evaluate_half_right(self, run_eval):
        entry = self.yaml_data[2]
        expected_score = float(entry["score"])

        score = float(run_eval["data"][2].get("score"))
        scores.append(score)

        assert abs(score - expected_score) <= THRESHOLD

    # def test_percentiles(self, init_eval):

    #     p20 = calculate_percentiles(scores, 20)
    #     p90 = calculate_percentiles(scores, 90)

    #     expected_p20 = 0.2
    #     expected_p90 = 0.9

    #     threshold = 0.05

    #     assert abs(p20 - expected_p20) <= threshold
    #     assert abs(p90 - expected_p90) <= threshold
