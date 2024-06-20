from .eval_prompt import EvalPromptTemplate

ANSWER_CORRECTNESS_BASE_TEMPLATE = """
You are given a question, an answer and reference text. You have to analyze each statement and classify them in one of the following categories:

- TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in reference,
- FP (false positive): statements present in the answer but not directly supported by any statement in reference,
- FN (false negative): statements found in the reference but not present in answer.

Each statement can only belong to one of the categories.

Your answer must strictly contain ONLY a valid JSON object representing the requested data. Your response must follow exactly the following JSON schema:
```
{{
  "TP": ["statement_1", "statement_2"],
  "FP": ["statement_3", "statement_4"],
  "FN": ["statement_5", "statement_6"]
}}
```

Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {query}
    ************
    [Reference]: {reference}
    ************
    [Answer]: {output}
    [END DATA]

Provide a valid JSON object based on previous instructions WITHOUT any additional characters!
"""

ANSWER_CORRECTNESS_TEMPLATE = EvalPromptTemplate(
    name="AnswerCorrectness",
    template=ANSWER_CORRECTNESS_BASE_TEMPLATE,
)
