FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    python3.11-dev \
    libsndfile1 \
    ffmpeg \
    tzdata \
    curl \
    build-essential \
    libpq-dev \
    espeak-ng \
    espeak-ng-data \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.11 /usr/bin/pip

RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

WORKDIR /app

COPY . .

RUN pip install --upgrade pip setuptools

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python3.11 -m spacy download en_core_web_sm

COPY .env .

EXPOSE 8000

CMD ["uvicorn", "back.api_main:app", "--host", "0.0.0.0", "--port", "8000"]