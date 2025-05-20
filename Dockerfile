# ───── Dockerfile ─────────────────────────────────────────────────────
FROM python:3.11-slim

# 1. system deps for numpy / scipy wheels (otherwise they compile & need RAM)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ gfortran libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. copy code (but NOT 11-million-row CSVs)
COPY app/ ./app

# Tell Uvicorn to listen on Railway’s provided $PORT
ENV PORT 8000
EXPOSE 8000

CMD ["uvicorn", "app.recommender_app:create_app", "--host", "0.0.0.0", "--port", "8000"]
