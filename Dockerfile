FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt ./
COPY requirements-dev.txt ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory
RUN mkdir -p /app/data/chroma

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
