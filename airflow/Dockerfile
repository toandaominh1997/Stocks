FROM apache/airflow:2.2.4-python3.8

USER root
RUN apt-get update && \
        apt-get install -y gcc make && \
        apt-get install -y g++ && \
        apt-get install -y cmake && \
        apt-get install -y git pandoc && \
        apt-get install -y gettext-base &&\
        apt-get install -y pkg-config libhdf5-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
RUN useradd -m tonne && echo "tonne:tonne" | chpasswd && adduser tonne sudo
USER tonne
WORKDIR /code
RUN python --version
RUN pip --version
RUN pip install --upgrade pip
RUN pip install jupyterlab
#RUN pip install --upgrade ipython ipykernel
#RUN ipython kernel install --name "python"
COPY requirements.txt /code
RUN cd /code && pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/phamdinhkhanh/vnquant.git@master

RUN pip install seaborn matplotlib plotly tqdm
RUN pip install Unidecode

