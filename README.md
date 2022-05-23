# Deploying your ML model
## From notebook to production with FastAPI and Docker

This repository is for the Vantage AI ML model productionization workshop.

## Instructions
To run the Docker container that hosts the FastAPI end-points, follow these instructions: 
- Clone this repository
- Make sure the Docker daemon is running
- Build the container image through copying the following command in your terminal: 
`docker build -t docker_workshop/csgo_prediction_service .
`
- Run the container with the following command: 
`docker run -p 8010:8000 -t docker_workshop/csgo_prediction_service
`
- Bonus: if you think those two commands are still to much to deploy your container, you can also use the Makefile and only run the `make docker` command

Et voilà! Your end-points should now be approachable via http://0.0.0.0:8010/. 

Tip: visit http://0.0.0.0:8010/docs to make use of the Swagger UI for FastAPI and test your end-points!

## Come join us!
Interested to work [with us](https://www.vantage-ai.com/en/team)? Take a look at our vacancies!

