FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libatlas-base-dev \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/amazon-science/chronos-forecasting.git

# Copy application
COPY nl-powerpredict-v3.py .
COPY templates/ templates/
COPY static/ static/

# Create directories
RUN mkdir -p data logs config

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "nl-powerpredict-v3.py"]
