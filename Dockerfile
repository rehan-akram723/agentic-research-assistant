# Use a lightweight, official Python runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc files to disk and ensure stdout/stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Ensure pip doesn't complain about root warnings during build
    PIP_ROOT_USER_ACTION=ignore

# Create a non-root user and group for security
RUN groupadd -r appgroup && useradd -r -g appgroup -s /sbin/nologin appuser

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for compiling certain ML/C++ libraries if wheels are absent)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Change ownership of the application directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Expose the ports for documentation purposes (Compose handles actual mapping)
EXPOSE 8000 8501

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]