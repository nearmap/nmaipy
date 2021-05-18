FROM jupyter/minimal-notebook

COPY ./requirements.txt /home/jovyan/nearmap-ai-user-guides/requirements.txt
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install -r requirements.txt

COPY ./ /home/jovyan/nearmap-ai-user-guides
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install .
