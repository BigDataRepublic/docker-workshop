FROM python:3.8-slim-buster

WORKDIR /app

# fix for libgomp missing (needed for lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

#CMD ["uvicorn", "main:app"]
CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]

