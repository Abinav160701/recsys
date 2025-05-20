# Start from an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install any Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code into the container
COPY app/ ./app

# Expose the port that FastAPI will run on (FastAPI's default is 8000)
EXPOSE 8000

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "app.recommender_app:app", "--host", "0.0.0.0", "--port", "8000"]
