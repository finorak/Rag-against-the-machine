export OLLAMA_MODELS=$(HOME)/goinfre/.models

run:
	@nohup ollama serve > ollama.log 2>&1 &

install:
	uv sync
	ollama pull "qwen3:0.6b"

debug:
	uv run python -m pdb src/__main__.py

flake:
	uv run python -m flake8 src

lint: flake
	uv run python -m mypy src

clean:
	find . -name "*.pyc" -exec rm -rf {} +
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +

fclean: clean
	rm -rf .venv

.PHONY := run debug install lint clean fclean
