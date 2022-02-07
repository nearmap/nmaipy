FROM jupyter/minimal-notebook

#ENV VIRTUAL_ENV=/home/jovyan/nearmap-ai-user-guides/.venv
#RUN python -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"
#
#RUN pip install --upgrade pip
#RUN which pip

#COPY ./requirements.txt /home/jovyan/nearmap-ai-user-guides/requirements.txt
#RUN cd /home/jovyan/nearmap-ai-user-guides \
# && pip install -r requirements.txt

COPY ./ /home/jovyan/nearmap-ai-user-guides
RUN cd /home/jovyan/nearmap-ai-user-guides \
 && pip install -r requirements.txt \
 && pip install .
