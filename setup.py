#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "httpx>=0.27.0",
    "nltk>=3.8.1",
    "numpy>=2.0.0",
    "python_Levenshtein>=0.25.1",
    "tiktoken>=0.7.0",
    "openai>=1.35.1",
]

test_requirements = ["pytest>=3", "PyYAML>=6.0.1", "numpy>=2.0.0"]

setup(
    author="Lynxius Inc.",
    author_email="contact@lynxius.ai",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.12",
    ],
    description="Lynxius speeds up LLM evaluation and prevents errors before they reach users.",
    long_description=readme,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "lynxius=lynxius.cli:main",
        ],
    },
    install_requires=requirements,
    include_package_data=True,
    keywords="lynxius",
    name="lynxius",
    packages=find_packages(
        include=["lynxius", "lynxius.*", "lynxius_evals", "lynxius_evals.*"]
    ),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={
        "test": test_requirements,
    },
    url="https://github.com/lynxius/lynxius-python",
    version="1.2.0",
    zip_safe=False,
)
