#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -v ~/Nenakhov/ros_ws:/ws \
    -it \
    ivan/cv 