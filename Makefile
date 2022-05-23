.PHONY: docker

docker:
	docker build -t docker_workshop/csgo_prediction_service .
	docker run -p 8000:8000 -t docker_workshop/csgo_prediction_service
