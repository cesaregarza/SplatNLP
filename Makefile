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
	-e DO_SPACES_ML_DIR="dataset_v2" \
	splatnlp:latest