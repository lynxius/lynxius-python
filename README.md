# Lynxius Python API Library

[![PyPI version](https://img.shields.io/pypi/v/lynxius.svg)](https://pypi.org/project/lynxius/)

[Lynxius](https://www.lynxius.ai/) speeds up LLM evaluation and prevents errors before they reach users.

[Lynxius Platform](https://platform.lynxius.ai/) offers advanced AI observability features, security, scalability, integrations with existing ecosystems and easy collaboration across teams.

The Lynxius Python library provides convenient access to the Lynxius REST API. The library includes type definitions for all request params and response fields, and offers synchronous clients powered by [httpx](https://github.com/encode/httpx).

## Docs

Start your LLM evaluation journey by consulting our official documentation at [docs.lynxius.ai](https://docs.lynxius.ai/).

## Lynxius Platform in 3 Minutes

Watch our 3-minute demo video to understand the key concepts of LLM evaluation.

[![Video Thumbnail](https://github-public-assets.s3.us-west-1.amazonaws.com/chatdoctorv2_datasetv2labeled.png)](https://github-public-assets.s3.us-west-1.amazonaws.com/Lynxius+Demo.mp4)



## Code Examples

Explore our tutorials:

1. [Lynxius to evaluate LLM Summarization and Custom Metrics](./tutorials/AI_medical_scribe_with_UI.ipynb)
2. [Lynxius to evaluate LLM chatbot applications](./tutorials/ChatDoctor.ipynb)
3. [Lynxius to boost collaboration with Subject Matter Experts](./tutorials/Datasets.ipynb)


## Create Development Environments

For local development, start by installing python `3.12.1`, creating a virtual environment and installing the dependencies:

```bash
python3.12 -m venv .lynxius-python
source .lynxius-python/bin/activate
pip install -r requirements.txt
```

## Contributing Guidelines

Would love to contribute? Please follows our [contribution guidelines](CONTRIBUTING.md).