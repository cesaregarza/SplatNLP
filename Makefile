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
