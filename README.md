# Turing Framework Testbed

## Requirements
* Docker or Docker Desktop installed
* Docker compose enabled

## Deployment
```bash
docker compose up --build -d
docker exec ollama ollama pull llama3.2
docker exec ollama ollama list
docker exec ollama ollama run llama3.2 "warm up"
```

## Usage
Access the Turing Framework through a web browser on the address `http://localhost:8080/`

MinIO is used for file storage backend, it is accessible on the address `http://localhost:9001/` 