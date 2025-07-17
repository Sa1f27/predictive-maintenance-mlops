# Dockerfile - Simple version for student project
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install -e .

# Create artifacts directory
RUN mkdir -p artifacts logs

# Expose port
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command
CMD ["python", "app.py"]