FROM ubuntu:18.04

# install required ubuntu packages
RUN apt-get update --fix-missing && \
    apt-get install -y libxrender1 libxext6 wget git

# install miniconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# add conda bin to path
ENV PATH /opt/conda/bin:$PATH

# use the environment.yml to create the conda env
COPY model_train/environment.yml /tmp/environment.yml

# create the conda env
RUN conda env create -n mmv_train -f /tmp/environment.yml

# activate env (add conda env bin to path)
ENV PATH /opt/conda/envs/mmv_train/bin:$PATH

# run the python script
WORKDIR /model_train
CMD ["python","-u","train_model.py"]
