FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --timeout 120 --retries 10 -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

USER vscode
