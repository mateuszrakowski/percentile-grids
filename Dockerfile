# Use Ubuntu as base image for better R support
FROM rocker/r-ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

RUN add-apt-repository ppa:deadsnakes/ppa -y

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-pip \
    python3.10-dev \
    python3.10-venv \
    build-essential \
    gfortran \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks to make python3.10 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.10 1

# Create and activate a virtual environment using Python 3.10
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install R packages
RUN R -e "install.packages(c('gamlss', 'gamlss.dist'), repos='https://cran.rstudio.com/')"

# Set working directory
WORKDIR /app

# Copy application files (you'll need to add your app files)
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Disable file watching
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

RUN mkdir -p /data /data/models

# Command to run Streamlit app
CMD ["streamlit", "run", "grids/main.py", "--server.port=8080", "--server.address=0.0.0.0"]