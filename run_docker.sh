#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -v /home/iiwa/Nenakhov:/iiwa \
    -it \
    ivan/cv 