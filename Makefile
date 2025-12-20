PYTHON ?= python
VENV ?= .venv

activate:
	@echo "source $(VENV)/bin/activate"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -e .

format:
	$(VENV)/bin/black src tests
	$(VENV)/bin/isort src tests

lint:
	PYTHONPATH=src $(VENV)/bin/ruff check src tests
	PYTHONPATH=src $(VENV)/bin/mypy src tests

test:
	PYTHONPATH=src $(VENV)/bin/pytest

check: lint test

.PHONY: activate venv install format lint test check
