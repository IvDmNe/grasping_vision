#! /bin/bash

sudo docker system prune -a -y
# sudo docker image prune
sudo docker rm $(sudo docker ps -a -f status=exited -q)