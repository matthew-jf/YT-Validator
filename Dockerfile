FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment file and create conda env
COPY environment-no-builds.yml .
RUN conda env create -f environment-no-builds.yml

# Activate environment and ensure it's used
SHELL ["conda", "run", "-n", "YT-Validator", "/bin/bash", "-c"]

# Copy application files
COPY app.py pipeline.py ./
COPY data ./data/

# Create data directory for runtime
RUN mkdir -p /app/data

# Set environment variables
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_DEBUG=0

# Cloud Run requires the app to listen on $PORT
CMD ["conda", "run", "-n", "YT-Validator", "python", "app.py"]