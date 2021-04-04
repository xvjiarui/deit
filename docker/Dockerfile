#FROM nvidia/cuda:11.1-cudnn8-devel
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /workspace

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    g++ \
    wget \
    bzip2 \
    git \
    vim \
    tmux \
    htop \
    git \
    zip \
    unzip \
    ca-certificates \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    libglu1-mesa \
    libxext6 \
    libxtst6 \
    libxrender1 \
    libxi6 \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswresample-dev \
    less \
    groff

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl https://rclone.org/install.sh | bash

# Set timezone
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime

# Get Conda-ified Python.
#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    sh ~/miniconda.sh -b -p /opt/conda && \
#    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install pytorch
#RUN conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge -y
RUN conda install -c conda-forge -c iopath -c fvcore fvcore -y
RUN conda install -c iopath iopath -y

# Install TIMM
RUN pip install timm==0.3.2

# Install zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -t robbyrussell -p git

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
