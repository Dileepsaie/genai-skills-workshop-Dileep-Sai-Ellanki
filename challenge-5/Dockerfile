FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py rag_service.py ./

ENV PORT=8081
CMD exec gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT} main:app
