FROM jupyter/minimal-notebook

USER root
RUN apt-get update && apt-get install -y libspatialindex-dev


USER jovyan

# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html?highlight=version#add-a-custom-conda-environment-and-jupyter-kernel
ARG conda_env=python39

COPY ./environment.yaml /home/${NB_USER}/tmp/environment.yaml
RUN conda config --set channel_priority flexible
RUN cd "/home/${NB_USER}/tmp/" && \
 mamba env create -p "${CONDA_DIR}/envs/${conda_env}" -f environment.yaml && \
 mamba clean --all -f -y

RUN mamba install ipykernel && \
"${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name="${conda_env}" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN echo "conda activate ${conda_env}" >> "${HOME}/.bashrc"