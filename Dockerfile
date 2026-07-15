FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment file and create conda env
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment and ensure it's used
SHELL ["conda", "run", "-n", "YT-Validator", "/bin/bash", "-c"]

# Copy application files and lookup data
COPY app.py pipeline.py helpers.py Licensed.csv assets_single_media_component.csv ./

# Pretrained model artifacts and shared feature engineering
COPY trey_pipeline/ml_pipeline ./trey_pipeline/ml_pipeline/
COPY trey_pipeline/models ./trey_pipeline/models/

# Create data directory for runtime
RUN mkdir -p /app/data

# Set environment variables
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_DEBUG=0

# Cloud Run requires the app to listen on $PORT
CMD ["conda", "run", "-n", "YT-Validator", "python", "app.py"]
