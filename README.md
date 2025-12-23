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

## Configuration
All configuration is performed using environment variables in `docker-compose.yml`.

### MinIO
* MINIO_ROOT_USER: The admin username for MinIO
* MINIO_ROOT_PASSWORD: The admin password for MinIO

### Ollama
* OLLAMA_HOST: The address that ollama backend listens on.
* OLLAMA_MODEL: The LLM model used by ollama backend.

### API Server
* OLLAMA_BASE_URL: The base URL to forward ollama requests to.
* OLLAMA_MODEL: The model used by ollama backend.
* NUM_PREDICT: The limit of tokens that the LLM is supposed to provide.
* MAX_FILE_CHARS: The maximum characters included in an uploaded file.
* MAX_UPLOAD_MB: The size limit for uploaded files.
* STORAGE_BACKEND: Either minio or local, switch that controls where the uploaded files are stored.
* MINIO_ENDPOINT: The MinIO endpoint to use for file storage.
* MINIO_ACCESS_KEY: The MinIO username to access the file storage.
* MINIO_SECRET_KEY: The MinIO password to access the file storage.
* MINIO_BUCKET: The name of the bucket for uploaded files in the MinIO backend.
* MINIO_SECURE: Enable ssh for MinIO access.