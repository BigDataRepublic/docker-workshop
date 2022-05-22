# docker-workshop
Private repository for the Vantage AI ML model productionization workshop.

To run the Docker container that hosts the FastAPI end-points, follow these instructions: 
- Make sure the Docker daemon is running
- Build the container image through copying the following command in your terminal: `	docker build -t docker_workshop/csgo_prediction_service .
`
- Run the container with the following command: `docker run -p 8010:8000 -t docker_workshop/csgo_prediction_service
`

Et voilà! Your end-points should now be approachable via http://0.0.0.0:8010/. 

Tip: visit http://0.0.0.0:8010/docs to make use of the Swagger UI for FastAPI and test your end-points!
