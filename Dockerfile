FROM python:3.8-slim-buster

WORKDIR /app

# libgomp1 is needed for lightgbm
RUN apt-get update && apt-get install libgomp1

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
