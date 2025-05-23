# Start from an official lightweight Python image
FROM python:3.11-slim

# Make a non-root user (Railway & Render best-practice)
RUN useradd -m recsys
WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ ./app
# Copy small static assets (CSV ≤ 1 MB)
COPY skus_metadata.csv /assets/
# OPTIONAL: if you sometimes ship boost_table.csv
#COPY boost_table.csv*  /assets/     
# * means “copy if exists”

ENV PORT=8080
EXPOSE 8080
USER recsys
ENV REDIS_URL="redis://default:caGsThrOKxjMqBtfAJsJIsDDnbZDQWxf@shortline.proxy.rlwy.net:43464"

#CMD ["uvicorn", "app.recommender_app:app", "--host", "0.0.0.0", "--port", "80"]

CMD ["sh", "-c", "uvicorn app.recommender_app:app --host 0.0.0.0 --port ${PORT}"]
