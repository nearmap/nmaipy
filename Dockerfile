FROM jupyter/minimal-notebook:1a66dd36ff82

USER root
RUN apt-get update && apt-get install -y libspatialindex-dev

USER jovyan
COPY ./requirements.txt /home/jovyan/nearmap-ai-user-guides/requirements.txt
ENV CONDA_VERSION=4.11.0
ENV MAMBA_VERSION=0.21.2

RUN sed -i '/^conda / d' $CONDA_DIR/conda-meta/pinned \
 && echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned

RUN conda install -c conda-forge -y conda=${CONDA_VERSION} mamba=${MAMBA_VERSION}

RUN cd /home/jovyan/nearmap-ai-user-guides \
 && conda config --set channel_priority flexible \
 && mamba env update --file environment.yaml

COPY ./ /home/jovyan/nearmap-ai-user-guides
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install .
