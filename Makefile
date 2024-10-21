.PHONY: build
build:
	docker rmi splatnlp:test_model_serve || true
	docker build \
		-t splatnlp:test_model_serve \
		.

.PHONY: run
run:
	docker run \
	-p 9000:9000 \
	-e VOCAB_URL="https://splat-nlp.nyc3.digitaloceanspaces.com/dataset_v2/vocab.json" \
	-e WEAPON_VOCAB_URL="https://splat-nlp.nyc3.digitaloceanspaces.com/dataset_v2/weapon_vocab.json" \
	-e MODEL_URL="https://splat-nlp.nyc3.digitaloceanspaces.com/dataset_v2/model.pth" \
	-e PARAMS_URL="https://splat-nlp.nyc3.digitaloceanspaces.com/dataset_v2/model_params.json" \
	splatnlp:test_model_serve