.PHONY: image build install test lint format clean

image: Dockerfile
	docker buildx build -t cognitively-guided-reasoning .

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

check-ollama:
	curl $(OLLAMA_API_BASE)/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ \
    "model": "qwen/qwen3-coder-30b", \
    "messages": [ \
      { "role": "system", "content": "Always answer in rhymes. Today is Thursday" }, \
      { "role": "user", "content": "What day is it today?" } \
    ], \
    "temperature": 0.7, \
    "max_tokens": -1, \
    "stream": false \
}'

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
