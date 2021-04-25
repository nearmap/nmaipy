FROM jupyter/minimal-notebook

COPY ./ /home/jovyan/nearmap-ai-user-guides

RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install -r requirements.txt \
 && pip install .
