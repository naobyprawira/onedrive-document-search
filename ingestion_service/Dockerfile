FROM python:3.10-slim

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install build tools and dependencies for pdf text extraction
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification first to leverage Docker cache
COPY requirements.txt ./requirements.txt
# Upgrade pip because the base image ships an older version that mis-hashes some wheels.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the default port for the FastAPI application (not strictly necessary here,
# since this service rarely exposes HTTP endpoints, but can be used for health checks)
EXPOSE 8000

# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
