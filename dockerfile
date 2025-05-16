###############################
#         Base Image          #
###############################
ARG BASE_IMAGE=python:3.11-slim

FROM $BASE_IMAGE AS base

WORKDIR /app

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1
ENV PATH="$PATH:$POETRY_HOME/bin"

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - 

RUN poetry config virtualenvs.create false

###############################
#    Install  Dependencies    #
###############################
FROM base AS dependencies

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-dev --without embeddings

###############################
#           Tests             #
###############################
FROM dependencies AS tests

COPY pyproject.toml poetry.lock ./
RUN poetry install --with dev --no-root

COPY src/splatnlp /app/src/splatnlp
COPY tests /app/tests

CMD ["pytest", "-q"]

###############################
#        Build Image          #
###############################
FROM dependencies AS build

ARG BUILD_VERSION

COPY README.md /app/
COPY pyproject.toml poetry.lock /app/
COPY src/splatnlp /app/src/splatnlp

# Build the application
RUN poetry version $BUILD_VERSION && \
    poetry build && \
    poetry install --no-dev --without embeddings && \
    poetry update

CMD ["poetry", "run", "uvicorn", "splatnlp.serve.app:app", "--host", "0.0.0.0", \
    "--port", "9000"]
