FROM jupyter/minimal-notebook

ENV VIRTUAL_ENV=/home/jovyan/nearmap-ai-user-guides
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./requirements.txt /home/jovyan/nearmap-ai-user-guides/requirements.txt
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install -r requirements.txt

COPY ./ /home/jovyan/nearmap-ai-user-guides
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install .
