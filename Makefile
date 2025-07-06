# Makefile – GNU make ≥ 4.0
.ONESHELL:
SHELL := /usr/bin/env bash

# ------------------------------------------------------------------
# GLOBALS (exported to every recipe)
# ------------------------------------------------------------------
export PATH := $(HOME)/.local/bin:$(PATH)  # so "poetry" is always visible
ACCEL ?= auto                             # override with `make install ACCEL=gpu`
CUDA_INDEX  := https://download.pytorch.org/whl/nightly/cu128
# Pre‑baked wheels known to work on B200 / CUDA 12.8
TORCH_URL   := $(CUDA_INDEX)/torch-2.8.0.dev20250623%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
VISION_URL  := $(CUDA_INDEX)/torchvision-0.23.0.dev20250623%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
AUDIO_URL   := $(CUDA_INDEX)/torchaudio-2.8.0.dev20250623%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl

.PHONY: build run build-upload install

# ------------------------------------------------------------------
# Simple Docker helpers (unchanged)
# ------------------------------------------------------------------
build:
	docker rmi splatnlp:latest || true
	docker build -t splatnlp:latest .

run:
	docker run \
		-p 9000:9000 \
		-e DO_SPACES_ML_ENDPOINT="https://splat-nlp.nyc3.digitaloceanspaces.com" \
		-e DO_SPACES_ML_DIR="dataset_v3" \
		splatnlp:latest

build-upload:
	docker build -t registry.digitalocean.com/sendouq/splatnlp:latest .
	doctl registry login
	docker push registry.digitalocean.com/sendouq/splatnlp:latest

# ------------------------------------------------------------------
# One‑shot installer – reproduces your manual script
# ------------------------------------------------------------------
install:
	# 0) Minimal OS packages (only once per host)
	sudo apt-get update -qq
	sudo apt-get install -y curl gcc python3-venv
	sudo rm -rf /var/lib/apt/lists/*

	# 1) Poetry itself
	if ! command -v poetry >/dev/null; then
		curl -sSL https://install.python-poetry.org | python3 -
	fi

	# 2) Always keep project venv **inside** the repo
	poetry config virtualenvs.in-project true

	# 3) Create .venv (so CPU & GPU builds can coexist on one VM)
	python3 -m venv .venv
	. .venv/bin/activate
	python -m pip install --upgrade pip

	# 4) Decide accelerator
	case "$(ACCEL)" in
		auto)
			if command -v nvidia-smi >/dev/null; then ACCEL=gpu; else ACCEL=cpu; fi ;;
		cpu|gpu)
			: ;;  # already valid
		*)
			echo "ACCEL must be auto|cpu|gpu"; exit 1 ;;
	esac
	echo "Installing for ACCEL=$$ACCEL …"

	# 5) Extra index only on GPU boxes
	if [[ "$$ACCEL" == "gpu" ]]; then
		export PIP_EXTRA_INDEX_URL="$(CUDA_INDEX)"
	fi

	# 6) Base dependencies from pyproject.toml
	poetry install --no-root --sync

	# 7) Swap in nightly wheels on GPUs
	if [[ "$$ACCEL" == "gpu" ]]; then
		echo "Replacing stable Torch with nightly CUDA 12.8 build"
		poetry run python -m pip uninstall -y torch torchvision torchaudio || true
		poetry run python -m pip install --no-cache-dir \
			"$(TORCH_URL)" \
			"$(VISION_URL)" \
			"$(AUDIO_URL)"
	fi

	# 8) Success banner
	echo
	echo "✅  Environment ready – activate with:  source .venv/bin/activate"
