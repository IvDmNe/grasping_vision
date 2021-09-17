#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -v ~/ros_ws:/ws \
    -v /dev:/dev \
    -it \
    --privileged \
    ivan/cv 