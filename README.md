# nearmap-ai-user-guides
A set of customer facing user guides to help them get started with Nearmap AI content, and inspire them with use cases.

docker build -t nearmap_ai .

docker run -it --rm \
  --name nearmap_ai2 \
  --volume ${PWD}/:/home/jovyan/nearmap-ai-user-guides \
  --volume ${PWD}/../data:/home/jovyan/data \
  --env API_KEY=${API_KEY} \
  --env NB_UID=$(id -u) \
  --env NB_GID=$(id -g) \
  --env GRANT_SUDO=yes \
  --user=root \
  -p 8889:8888 \
  nearmap_ai \
  start.sh bash

jupyter notebook --port=8888 --no-browser --ip=* --allow-root