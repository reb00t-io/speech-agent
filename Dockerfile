FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml VERSION ./
COPY config/ ./config/
COPY docs/ ./docs/
COPY src/ .
RUN pip install --no-cache-dir .
ARG DEPLOY_DATE=unknown
ENV DEPLOY_DATE=$DEPLOY_DATE
ARG PORT
ENV PORT=$PORT

RUN useradd --create-home appuser
USER appuser

EXPOSE $PORT
CMD ["python", "main.py"]
