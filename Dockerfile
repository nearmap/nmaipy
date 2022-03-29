FROM jupyter/minimal-notebook:4d9c9bd9ced0

COPY ./requirements.txt /home/jovyan/nearmap-ai-user-guides/requirements.txt
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install -r requirements.txt

COPY ./ /home/jovyan/nearmap-ai-user-guides
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install .
