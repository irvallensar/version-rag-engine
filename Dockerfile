# Use the slim Python 3.12 image to keep the container lightweight
FROM python:3.12-slim

# Copy the uv binary directly from Astral's official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy only the dependency files first to cache the installation layer
COPY pyproject.toml uv.lock* ./

# Install dependencies into the container's system environment
# This avoids creating a nested virtual environment inside Docker
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy the actual application code
COPY src/ ./src/

# Expose the port FastAPI will run on
EXPOSE 8000

# Set the startup command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]