FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml VERSION ./
COPY config/ ./config/
COPY docs/ ./docs/
COPY src/ .
# git is needed to install the memorizer dependency (pinned git+https release);
# installed and removed in one layer to keep the image slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir . \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
ARG DEPLOY_DATE=unknown
ENV DEPLOY_DATE=$DEPLOY_DATE
ARG PORT
ENV PORT=$PORT

RUN useradd --create-home appuser
USER appuser

EXPOSE $PORT
CMD ["python", "main.py"]
