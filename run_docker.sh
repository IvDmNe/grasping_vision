#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -it \
    ivan/cv 