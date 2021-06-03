# nearmap-ai-user-guides
A set of customer facing user guides to help them get started with Nearmap AI content, and inspire them with use cases.

## Run with Docker

Using a command line interface, navigate to the project root directory. Build the docker image with:
```
docker build -t nearmap_ai .
```

Once the build has complete, start the image with (note: your data directory may have a local different path):
```
docker run -it --rm \
  --name nearmap_ai \
  --volume ${PWD}/:/home/jovyan/nearmap-ai-user-guides \
  --volume ${PWD}/../data:/home/jovyan/data \
  --env API_KEY=${API_KEY} \
  --env NB_UID=$(id -u) \
  --env NB_GID=$(id -g) \
  --env GRANT_SUDO=yes \
  --user=root \
  -p 8888:8888 \
  nearmap_ai \
  start.sh bash
```
Start a notebook server in the Docker container:
```
jupyter notebook --port=8888 --no-browser --ip=* --allow-root
```
You should now be able to access the notebook server at `localhost:8888` (get the token in the notebook logs).

To run batches you can use docker-compose. Update the variables in `docker-compose.yaml` and run:

```
export NB_UID=$(id -u) && export NB_GID=$(id -g) && docker-compose build && docker-compose up --remove-orphans
```