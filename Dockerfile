FROM ubuntu:18.04

# install required ubuntu packages
RUN apt-get update --fix-missing
RUN apt-get install -y libxrender1 libxext6 wget git

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

# clone and install modifield naive bayes model
RUN git clone https://github.com/chembl/ModifiedNB.git /tmp/ModifiedNB
WORKDIR /tmp/ModifiedNB
RUN python setup.py install

# run the python script
WORKDIR /model_train
CMD python train_model.py
