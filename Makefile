.DEFAULT_GOAL := help
SHELL := /bin/bash


.PHONY: install ## install required dependencies on bare metal
install:
	uv sync --refresh


.PHONY: format ## Run the formatter on bare metal
format:
	uv run ruff format
	uv run ruff check --fix


.PHONY: lint ## run the linter on bare metal
lint:
	uv run ruff check
	uv run ruff format --check


.PHONY: test ## run unit tests on bare metal
test:
	uv run pytest -v -m "not integration"
