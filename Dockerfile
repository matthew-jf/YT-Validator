FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment file and create conda env
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment and ensure it's used
SHELL ["conda", "run", "-n", "YT-Validator", "/bin/bash", "-c"]

# Copy application files and lookup data
COPY app.py pipeline.py helpers.py Licensed.csv assets_single_media_component.csv ./
COPY scripts ./scripts/

# Shared feature engineering + the export/verify tooling
COPY trey_pipeline/ml_pipeline ./trey_pipeline/ml_pipeline/

# The model artifact (~1.4 GB) is deliberately NOT copied from the build
# context. It is fetched from GCS by scripts/fetch_model.sh, because copying
# from the working tree is how unresolved Git LFS pointers (~130-byte text
# files) got baked into images: the build succeeded, the service started, and
# only the first prediction failed.
#
# Provide the model one of two ways:
#   1. build arg, baked in (immutable image, larger):
#        docker build --build-arg MODEL_BUCKET=gs://... --build-arg MODEL_VERSION=v1 .
#      with GCS credentials available to the build.
#   2. runtime mount (smaller image, model versioned independently):
#        docker run -v /opt/models:/app/trey_pipeline/models ...
ARG MODEL_BUCKET=""
ARG MODEL_VERSION="v1"
RUN mkdir -p /app/trey_pipeline/models && \
    if [ -n "$MODEL_BUCKET" ]; then \
        MODEL_BUCKET="$MODEL_BUCKET" MODEL_VERSION="$MODEL_VERSION" \
            bash scripts/fetch_model.sh; \
    else \
        echo "No MODEL_BUCKET at build time; mount the model at runtime."; \
    fi

# Create data directory for runtime
RUN mkdir -p /app/data

# Set environment variables
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_DEBUG=0
ENV FLASK_RUN_PORT=3001

# /health reports 503 until the model is loaded and warm, so it is a genuine
# readiness signal rather than a liveness stub.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:${FLASK_RUN_PORT}/health || exit 1

# --no-capture-output so the app's logs stream instead of being buffered by
# `conda run` until the process exits.
CMD ["conda", "run", "--no-capture-output", "-n", "YT-Validator", "python", "app.py"]
