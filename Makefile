setup-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	source .venv/bin/activate

install-dev:
	uv sync --dev
	uv run maturin develop

test:
	uv run pytest

jupyter-lab:
	uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=pyhgf
	uv run --with jupyter jupyter lab

pre-commit:
	uv run pre-commit install
	uv run pre-commit autoupdate
	uv run pre-commit run --all-files

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 						# running ruff formatting
	uv run ruff check --fix	--extend-select=I	# running ruff linting

run-all-notebooks:
	@echo "--- 📚 Running all notebooks ---"
	cd docs/source/notebooks/ && uv run python -m nbconvert *.ipynb --to notebook --execute --inplace

build-docs:
	@echo "--- 📖 Building docs ---"
	uv run sphinx-build -j 1 -T -b html docs/source docs/build/html