.PHONY: docker

docker:
	docker build -t docker_workshop/csgo_prediction_service .
	docker run docker_workshop/csgo_prediction_service
