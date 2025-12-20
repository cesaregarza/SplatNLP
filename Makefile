# Makefile  – GNU make ≥ 4.0
.ONESHELL:
SHELL := /usr/bin/env bash

.PHONY: build run build-upload install poetry-path venv python-deps

build:
	docker rmi splatnlp:latest || true
	docker build \
		-t splatnlp:latest \
		.

run:
	docker run \
		-p 9000:9000 \
		-e DO_SPACES_ML_ENDPOINT="https://splat-nlp.nyc3.digitaloceanspaces.com" \
		-e DO_SPACES_ML_DIR="dataset_v3" \
		splatnlp:latest

build-upload:
	docker build \
		-t registry.digitalocean.com/sendouq/splatnlp:latest \
		.
	doctl registry login
	docker push registry.digitalocean.com/sendouq/splatnlp:latest

# ---------------------------------------------------------------
# make install [ACCEL=auto|cpu|gpu]  ← defaults to auto-detect
# ---------------------------------------------------------------
ACCEL ?= auto
CUDA_INDEX := https://download.pytorch.org/whl/nightly/cu128
TORCH_NIGHTLY_PIN := 2.8.0.dev20250624+cu128  # ← the build you trust

install: poetry-path venv python-deps

# 1) make sure Poetry itself is on PATH *now* and next time
poetry-path:
	set -euo pipefail
	if ! command -v poetry >/dev/null; then
		curl -sSL https://install.python-poetry.org | python3 -
	fi
	export PATH="$$HOME/.local/bin:$$PATH"
	grep -qxF 'export PATH="$$HOME/.local/bin:$$PATH"' $$HOME/.bashrc \
	  || echo 'export PATH="$$HOME/.local/bin:$$PATH"' >> $$HOME/.bashrc

# 2) local venv so CPU & GPU installs can coexist on the same VM image
venv:
	python3 -m venv .venv
	. .venv/bin/activate
	python -m pip install --upgrade pip

# 3) the *only* place we touch Torch
python-deps:
	set -euo pipefail
	. .venv/bin/activate

	# -------- optional GPU auto-detect --------
	if [ "$(ACCEL)" = "auto" ]; then
		if command -v nvidia-smi >/dev/null; then ACCEL=gpu; else ACCEL=cpu; fi
	fi

		# -------- steer Poetry's pip via env var ----
	if [ "$$ACCEL" = "gpu" ]; then
		export PIP_EXTRA_INDEX_URL="$(CUDA_INDEX)"
	fi

	# -------- one-shot Poetry install ----------
	poetry install --no-root --sync

	# -------- pin nightly exactly on GPU boxes --
	if [ "$$ACCEL" = "gpu" ]; then
		poetry run python -m pip install --no-deps \
			torch=="$(TORCH_NIGHTLY_PIN)" \
			--index-url "$(CUDA_INDEX)"
	fi
