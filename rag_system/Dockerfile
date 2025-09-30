FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install pip -U
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/pdf data/indices

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "-m", "api.main"]