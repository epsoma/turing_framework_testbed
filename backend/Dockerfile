FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY static /app/static

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["gunicorn", "-w", "1", "--worker-class", "gthread", "--threads", "8", "-b", "0.0.0.0:8080", "--timeout", "600", "--graceful-timeout", "30", "app:app"]

