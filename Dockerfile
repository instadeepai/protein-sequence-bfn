ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    g++ \
    wget \
    unzip \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install Micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Install conda dependencies
COPY environment.yaml /tmp/environment.yaml

ARG ACCELERATOR

# Ensure the correct accelerator is installed.
ARG ACCELERATOR
RUN if [ "${ACCELERATOR}" = "GPU" ]; then \
    sed -i 's/jax==/jax[cuda12_pip]==/g' /tmp/environment.yaml && \
    sed -i 's/libtpu_releases\.html/jax_cuda_releases\.html/g' /tmp/environment.yaml;\
    echo "    - nvidia-cudnn-cu12==8.9.7.29" >> /tmp/environment.yaml; fi

RUN if [ "${ACCELERATOR}" = "TPU" ]; then \
    sed -i 's/jax==/jax[tpu]==/g' /tmp/environment.yaml; fi

RUN if [ "${ACCELERATOR}" = "CPU" ]; then \
    echo "Building for cpu" ; fi

RUN micromamba create -y --file /tmp/environment.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete
ENV PATH=/opt/conda/envs/protbfn/bin/:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/protbfn/lib/:$LD_LIBRARY_PATH

COPY . /app
# Create main working folder
# RUN mkdir /app
WORKDIR /app
RUN pip install -U "huggingface_hub[cli]"

# Disable debug, info, and warning tensorflow logs
ENV TF_CPP_MIN_LOG_LEVEL=3