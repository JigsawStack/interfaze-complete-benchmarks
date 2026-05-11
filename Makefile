.PHONY: init lint fmt check

init:
	uv sync

lint:
	uv run ruff check .
