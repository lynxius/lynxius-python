import os
import pytest
import yaml
import lynxius
import numpy as np

from lynxius.client import LynxiusClient
from lynxius.evals.answer_correctness import AnswerCorrectness

os.environ['LYNXIUS_API_KEY'] = 'nsAqR0hp7POahcIZYNL4c0g7ytonozWllE3102IeMEzauOlS'

api_key = os.getenv('LYNXIUS_API_KEY')

client = LynxiusClient(api_key=api_key)

scores = []

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


@pytest.fixture
def init_model():
    label = "iter_1"
    tags = ["answer_correctness"]
    return AnswerCorrectness(label=label, tags=tags)

def calculate_percentiles(scores, percentile):
    return np.percentile(scores, percentile)


@pytest.mark.usefixtures("yaml_data")
class TestAnswerCorrectness:
    """Test `AnswerCorrectness` evaluator."""

    @pytest.fixture
    def init_eval(self, init_model):
        return init_model

    def test_init(self, init_eval):
        assert init_eval.label == "iter_1"
        assert "answer_correctness" in init_eval.tags

    def test_evaluate_success(self, init_eval):
        entry = self.yaml_data[0]
        expected_score = float(entry["score"])
        threshold = 0.1

        init_eval.add_trace(query=entry["input"], reference=entry["reference"], output=entry["output"])
        answer_correctness_uuid = client.evaluate(init_eval)
        pr_eval_run = client.get_eval_run(answer_correctness_uuid)
        score = pr_eval_run.get("aggregate_score")
        scores.append(score)

        assert abs(score - expected_score) <= threshold

    def test_evaluate_failure(self, init_eval):
        entry = self.yaml_data[1]
        expected_score = float(entry["score"])
        threshold = 0.1

        init_eval.add_trace(query=entry["input"], reference=entry["reference"], output=entry["output"])
        answer_correctness_uuid = client.evaluate(init_eval)
        pr_eval_run = client.get_eval_run(answer_correctness_uuid)
        score = pr_eval_run.get("aggregate_score")
        scores.append(score)

        assert abs(score - expected_score) <= threshold

    def test_evaluate_half_right(self, init_eval):
        entry = self.yaml_data[2]
        expected_score = float(entry["score"])
        threshold = 0.1

        init_eval.add_trace(query=entry["input"], reference=entry["reference"], output=entry["output"])
        answer_correctness_uuid = client.evaluate(init_eval)
        pr_eval_run = client.get_eval_run(answer_correctness_uuid)
        score = pr_eval_run.get("aggregate_score")
        scores.append(score)

        assert abs(score - expected_score) <= threshold

    def test_percentiles(self, init_eval):

        p20 = calculate_percentiles(scores, 20)
        p90 = calculate_percentiles(scores, 90)

       
        expected_p20 = 0.2  
        expected_p90 = 0.9  

        threshold = 0.05 

        assert abs(p20 - expected_p20) <= threshold
        assert abs(p90 - expected_p90) <= threshold