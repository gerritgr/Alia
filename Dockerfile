FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
ARG username

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get install libxrender1


# Create a working directory
RUN mkdir /main
RUN mkdir /main/home
WORKDIR /main

# RUN git clone https://github.com/gerritgr/nextaid.git

# Create a non-root user and switch to it
#RUN adduser --disabled-password --gecos '' --shell /bin/bash $username \
#    && chown -R $username:$username /main
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER $username

# All users can use /home/user as their home directory
ENV HOME=/main/home
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME


RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Set up the Conda environment (using Miniforge)
ENV PATH=$HOME/mambaforge/bin:$PATH
COPY environment.yml /main/environment.yml
RUN conda env update -n base -f /main/environment.yml
#RUN curl -sLo ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
#    && chmod +x ~/mambaforge.sh \
#    && ~/mambaforge.sh -b -p ~/mambaforge \
#    && rm ~/mambaforge.sh \
#    && mamba env update -n base -f /main/environment.yml \
#    #&& rm /main/environment.yml \
#    && mamba clean -ya


RUN conda run -n base python -c "import torch_geometric; print('torch_geometric.__version__:', torch_geometric.__version__)"

RUN conda env export --no-builds

    
# install jax
#RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # should be don in env file
