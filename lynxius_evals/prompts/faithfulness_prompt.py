from .eval_prompt import EvalPromptTemplate

FAITHFULNESS_BASE_TEMPLATE = """
You are given a question, an answer, and reference text. You have to break the ANSWER into individual statmeents, analyze each statement, and classify it into one of the following categories:

- "faithful": if the statement CAN be directly inferred from the reference text,
- "unfaithful": if the statement CANNOT be directly inferred from the reference text.

Your answer must strictly contain ONLY a valid JSON object representing the requested data. Your response must follow exactly the following JSON schema:
```
{{
    "faithful": ["statement_1", "statement_2"],
    "unfaithful": ["statement_3", "statement_5"],
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

FAITHFULNESS_TEMPLATE = EvalPromptTemplate(
    name="Faithfulness",
    template=FAITHFULNESS_BASE_TEMPLATE,
)
