from .eval_prompt import EvalPromptTemplate

CONTEXT_PRECISION_BASE_TEMPLATE = """
You are given a question, an answer and a list of contexts that were used to answer the given question. You have to determine if each of the given contexts were relevant when answering the question.

You have to return a valid JSON object containing a list of integers that are either 0 or 1. The length of this list must match exactly the length of the contexts. This means that for every context you have to print `0` if it is not relevant when answering the question and `1` if it is relevant. You can not print any additional content apart from a valid JSON objects. This is an example of your output for the query that contains 3 contexts:
```
{{
  "result": [0, 1, 1]
}}
```

Here is the data:
[BEGIN DATA]
************
[Question]: {query}
************
[Answer]: {reference}
{context}
[END DATA]

Provide a valid JSON object based on previous instructions WITHOUT any additional characters!
"""

CONTEXT_PRECISION_TEMPLATE = EvalPromptTemplate(
    name="ContextPrecision",
    template=CONTEXT_PRECISION_BASE_TEMPLATE,
)
