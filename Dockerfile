FROM python:3.10-slim

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install build tools and dependencies
# Ingestion service needs: build-essential, libpq-dev, gcc, p7zip-full
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    p7zip-full \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY ingestion_service ./ingestion_service
COPY search_service ./search_service
COPY streamlit_app ./streamlit_app

# Copy logo to streamlit app directory
COPY logo.png ./streamlit_app/logo.png

# Copy start script
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# Expose ports
EXPOSE 8011 8012 8503

# Start all services
CMD ["./start.sh"]
