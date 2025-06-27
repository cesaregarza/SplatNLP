.PHONY: build
build:
	docker rmi splatnlp:latest || true
	docker build \
		-t splatnlp:latest \
		.

.PHONY: run
run:
	docker run \
	-p 9000:9000 \
	-e DO_SPACES_ML_ENDPOINT="https://splat-nlp.nyc3.digitaloceanspaces.com" \
	-e DO_SPACES_ML_DIR="dataset_v3" \
	splatnlp:latest

.PHONY: build-upload
build-upload:
	docker build \
		-t registry.digitalocean.com/sendouq/splatnlp:latest \
		.
	doctl registry login
	docker push registry.digitalocean.com/sendouq/splatnlp:latest

.PHONY: install
install:
	sudo apt-get update && sudo apt-get install -y \
		curl \
		gcc \
		python3-venv \
		&& sudo rm -rf /var/lib/apt/lists/*

	curl -sSL https://install.python-poetry.org | python3 -
	python3 -m venv .venv
	export PATH="/home/ubuntu/.local/bin:${PATH}"
	poetry config virtualenvs.in-project true
	poetry install
	export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/nightly/"
	. .venv/bin/activate
	# Remove torch to install nightly version for B200
	poetry remove torch
	export TORCH_URL="https://download.pytorch.org/whl/nightly/cu128/torch-2.8.0.dev20250623%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
	export VISION_URL="https://download.pytorch.org/whl/nightly/cu128/torchvision-0.23.0.dev20250623%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
	export AUDIO_URL="https://download.pytorch.org/whl/nightly/cu128/torchaudio-2.8.0.dev20250623%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
	pip install ${TORCH_URL} ${VISION_URL} ${AUDIO_URL}
