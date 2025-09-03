.PHONY: build install test lint format clean

build:
	sudo apt update
	sudo apt install -y coq
	coqc --version

install:
	pip install -e .[dev]

test:
	pytest

lint:
	ruff check .

format:
	black .
	isort .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
