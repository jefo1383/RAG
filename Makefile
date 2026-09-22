FLAKE8_EXCLUDE = --exclude=.venv,data,__pycache__
MYPY_EXCLUDE = --exclude '(\.venv|data|__pycache__)'
export UV_CACHE_DIR = /goinfre/$(USER)/.cache/uv
export HF_HOME = /goinfre/$(USER)/.cache/huggingface

.PHONY: install-42 install run debug clean fclean re lint lint-strict

install-42:
	@which uv > /dev/null 2>&1 || pip install --user uv
	mkdir -p /goinfre/$(USER)/RAG_env
    mkdir -p $(UV_CACHE_DIR)
	mkdir -p $(HF_HOME)
	uv venv /goinfre/$(USER)/RAG_env/.venv
	ln -sf /goinfre/$(USER)/RAG_env/.venv .venv
	uv sync

install:
	@which uv > /dev/null 2>&1 || pip install --user uv
	uv sync

run:
	uv run python -m src $(cmd)

debug:
	uv run python -m pdb src

clean:
	rm -rf __pycache__ .mypy_cache .python-version

fclean: clean
	rm -rf /goinfre/$(USER)/RAG_env
	rm -rf .venv

re: fclean install

lint:
	uv run flake8 src $(FLAKE8_EXCLUDE)
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src $(MYPY_EXCLUDE)

lint-strict:
	uv run flake8 src $(FLAKE8_EXCLUDE)
	uv run mypy --strict src $(MYPY_EXCLUDE)

doc:
	uv run pydocstyle src