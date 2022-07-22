export NB_UID=$(id -u) && export NB_GID=$(id -g) && docker-compose build && docker-compose --file docker-compose.yml up --remove-orphans;
