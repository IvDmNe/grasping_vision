#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -v ~/:/local \
    -it \
    ivan/cv 