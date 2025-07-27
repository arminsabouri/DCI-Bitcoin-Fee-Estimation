FROM python:3.8.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential binutils


WORKDIR /app
COPY src .
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pyinstaller --onefile data_lake/main.py
