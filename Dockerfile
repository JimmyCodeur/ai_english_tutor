# Use a base image compatible with CUDA and Python 3.11
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, pip, libsndfile, ffmpeg, and build-essential
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
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Create symbolic links for python and pip to point to the correct versions
RUN ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.11 /usr/bin/pip

# Set the time zone to UTC
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# Define the working directory in the container
WORKDIR /app

# Copy application files
COPY . .

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies in a controlled order to avoid conflicts
RUN pip install --no-cache-dir spacy==3.4.3
RUN pip install --no-cache-dir pydantic==1.10.2
RUN pip install --no-cache-dir fastapi==0.95.2 uvicorn==0.20.0 sqlalchemy==1.4.41
RUN pip install --no-cache-dir nltk==3.7 aiofiles==0.8.0 passlib==1.7.4 psycopg2-binary==2.9.3 python-dotenv==0.21.0 email-validator==1.2.1 ollama==0.1.0 psutil==5.9.1 python-jose==3.3.0
RUN pip install --no-cache-dir python-multipart

# Install TTS dependencies with compatible versions
RUN pip install --no-cache-dir TTS==0.15.6 faster-whisper==0.5.1 pydub==0.25.1 noisereduce==2.0.1 scipy==1.9.3 langdetect==1.0.9 langid==1.1.6
RUN pip install --no-cache-dir numpy thinc
RUN pip install --no-cache-dir soundfile>=0.12.1 librosa==0.10.0

# Download the spaCy model using python3.11
RUN python3.11 -m spacy download en_core_web_sm

# Copy the .env file
COPY .env .

# Expose the port the application will run on
EXPOSE 8000

# Define the default command to run the application
CMD ["uvicorn", "back.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
