# AGENT GUIDELINES

This repository uses **Poetry** for dependency management and testing.
When working on this project:

* Install dependencies with `poetry install --with dev`.
* Run the test suite using `poetry run pytest -q`.
* Apply code formatting with `poetry run black .` and sort imports with `poetry run isort .`.
  The configured line length for both tools is 80 characters (see `pyproject.toml`).
* The codebase targets Python 3.10+, and the CI tests run on Python 3.11.

Follow these steps before submitting changes.
