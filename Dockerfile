FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment file and create conda env
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment and ensure it's used
SHELL ["conda", "run", "-n", "YT-Validator", "/bin/bash", "-c"]

# Application code, plus the artifacts pipeline.py resolves next to itself
COPY app.py pipeline.py helpers.py ./
COPY model.joblib Licensed.csv assets_single_media_component.csv ./

# Runtime scratch: uploads, outputs and the task store live here
RUN mkdir -p /app/data

ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_DEBUG=0
EXPOSE 3001

CMD ["conda", "run", "--no-capture-output", "-n", "YT-Validator", "python", "app.py"]
