import os
import pytest # type: ignore
import yaml # type: ignore
import time
from lynxius.client import LynxiusClient # type: ignore
from lynxius.evals.context_precision import ContextPrecision # type: ignore
from lynxius.rag.types import ContextChunk # type: ignore

# Set the environment variable
os.environ['LYNXIUS_API_KEY'] = '<ADD LYNXIUS API KEY>'

# Now you can access the API key from the environment variable
api_key = os.getenv('LYNXIUS_API_KEY')

client = LynxiusClient(api_key=api_key)

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

@pytest.fixture
def init_model():
    label = "iter_1"
    tags = ["context_precision"]
    return ContextPrecision(label=label, tags=tags)

@pytest.mark.usefixtures("yaml_data")
class TestContextPrecision:
    """Test `ContextPrecision` evaluator."""

    @pytest.fixture
    def init_eval(self, init_model):
        return init_model

    def test_init(self, init_eval):
        assert init_eval.label == "iter_1"
        assert "context_precision" in init_eval.tags

    def test_yaml_data(self):
        # Access the YAML data attached to the class
        assert len(self.yaml_data['test_cases']) == 18

    def test_evaluate_success(self, init_eval):
        entry = self.yaml_data['test_cases'][0]
        expected_score = float(entry["expected_output"])
        threshold = 0.15
    
        init_eval.add_trace(
            query=entry["input"]["query"],
            reference=entry["input"]["reference"],
            context=[ContextChunk(document=ctx["document"], relevance=ctx["relevance"]) for ctx in entry["input"]["contexts"]]
        )
        result = client.evaluate(init_eval)
        print("Evaluation result for success case:", result)  # Debugging statement
    
        assert isinstance(result, str), f"Expected result to be a str (run ID), got {type(result)}"
        assert result is not None and len(result) > 0, "Run ID should not be None or empty"

    def test_evaluate_failure(self, init_eval):
        entry = self.yaml_data['test_cases'][1]
        expected_score = float(entry["expected_output"])
        threshold = 0.15
    
        init_eval.add_trace(
            query=entry["input"]["query"],
            reference=entry["input"]["reference"],
            context=[ContextChunk(document=ctx["document"], relevance=ctx["relevance"]) for ctx in entry["input"]["contexts"]]
        )
        result = client.evaluate(init_eval)
        print("Evaluation result for failure case:", result)  # Debugging statement
    
        assert isinstance(result, str), f"Expected result to be a str (run ID), got {type(result)}"
        assert result is not None and len(result) > 0, "Run ID should not be None or empty"

    def test_evaluate_partial_success(self, init_eval):
        entry = self.yaml_data['test_cases'][2]
        expected_score = float(entry["expected_output"])
        threshold = 0.15
    
        init_eval.add_trace(
            query=entry["input"]["query"],
            reference=entry["input"]["reference"],
            context=[ContextChunk(document=ctx["document"], relevance=ctx["relevance"]) for ctx in entry["input"]["contexts"]]
        )
        result = client.evaluate(init_eval)
        print("Evaluation result for partial success case:", result)  # Debugging statement
    
        assert isinstance(result, str), f"Expected result to be a str (run ID), got {type(result)}"
        assert result is not None and len(result) > 0, "Run ID should not be None or empty"