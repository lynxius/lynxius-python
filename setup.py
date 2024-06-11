#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#     history = history_file.read()

requirements = [
    # "Click>=7.0",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Lynxius Inc.",
    author_email="contact@lynxius.ai",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
    ],
    description="AI Observability & Evaluation - Evaluate, troubleshoot, and test your LLM models online and offline.",
    entry_points={
        "console_scripts": [
            "lynxius=lynxius.cli:main",
        ],
    },
    install_requires=requirements,
    # long_description=readme + "\n\n" + history,
    long_description=readme,
    include_package_data=True,
    keywords="lynxius",
    name="lynxius",
    packages=find_packages(include=["lynxius", "lynxius.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/lynxius/lynxius-python",
    version="0.1.1",
    zip_safe=False,
)
