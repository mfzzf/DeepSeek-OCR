# DeepSeek-OCR vLLM API Server Dockerfile
# Using official vLLM image as base (includes PyTorch, CUDA, and vLLM)
# Using CUDA 12.4 compatible version
FROM vllm/vllm-openai:v0.6.4.post1-cu124

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy requirements files first (for better caching)
COPY requirements.txt /app/
COPY DeepSeek-OCR-vllm/requirements_api.txt /app/

# Install additional dependencies
# vLLM, PyTorch, FastAPI, Uvicorn are already in the base image
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY DeepSeek-OCR-vllm/ /app/

# Make entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

# Create directory for models (can be mounted as volume)
RUN mkdir -p /app/models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set default environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    VLLM_USE_V1=1

# Use bash as entrypoint to run our script
ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]

