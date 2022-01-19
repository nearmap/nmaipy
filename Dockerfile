FROM jupyter/minimal-notebook

USER root
RUN apt-get update && apt-get install -y libspatialindex-dev

USER jovyan
COPY ./requirements.txt /home/jovyan/nearmap-ai-user-guides/requirements.txt
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install -r requirements.txt

COPY ./ /home/jovyan/nearmap-ai-user-guides
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install .
