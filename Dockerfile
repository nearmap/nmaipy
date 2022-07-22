FROM jupyter/minimal-notebook

USER root
RUN apt-get update && apt-get install -y libspatialindex-dev

USER jovyan
COPY ./environment.yaml /home/jovyan/nearmap-ai-user-guides/environment.yaml
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && conda config --set channel_priority flexible \
 && mamba env update --file environment.yaml

COPY ./ /home/jovyan/nearmap-ai-user-guides
USER root
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install .
USER jovyan