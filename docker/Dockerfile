
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    apt-utils \
    net-tools \
    mesa-utils \
    gnupg2 \
    wget \
    curl \
    git \
    mc \
    nano \
    cmake \
    cmake-curses-gui \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update && apt-get install -y ros-melodic-ros-base


RUN apt-get update && apt-get install -y ros-melodic-eigen-stl-containers
RUN apt-get update && apt-get install -y ros-melodic-control*
RUN apt-get update && apt-get install -y ros-melodic-random-numbers ros-melodic-resource-retriever \
    ros-melodic-shape-msgs ros-melodic-visualization-msgs

RUN apt-get update && apt-get install -y ros-melodic-moveit  ros-melodic-joint* ros-melodic-robot-state-publisher

RUN apt-get update && apt-get install -y ros-melodic-gazebo-ros
RUN apt-get update && apt-get install -y ros-melodic-camera-info-manager
RUN apt-get update && apt-get install -y ros-melodic-depth-image-proc


RUN apt-get install -y ros-melodic-visp* \
    ros-melodic-camera-info-manager* \
    ros-melodic-image-transport* \
    ros-melodic-codec-image-transport \
    ffmpeg

RUN apt-get install -y ros-melodic-usb-cam ros-melodic-image-view

RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc

# RUN apt-get install dialog apt-utils -y

# install miniconda
# RUN apt-get -qq -y install bzip2 \
#     && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
#     && bash /tmp/miniconda.sh -bfp /usr/local \
#     && rm -rf /tmp/miniconda.sh \
#     && conda install -y python=3 \
#     && conda update conda \
#     # && apt-get -qq -y remove curl bzip2 \
#     # && apt-get -qq -y autoremove \
#     # && apt-get autoclean \
#     && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
#     && conda clean --all --yes

# Anaconda installing
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN conda init

# clone project

COPY . /point_cloud_processing

RUN conda env create -f /point_cloud_processing/scripts/cv_env.yml

# RUN git clone https://github.com/IvDmNe/point_cloud_processing.git && \
# RUN conda env create -f point_cloud_processing/scripts/cv_env.yml

RUN echo "conda activate cv" >> ~/.bashrc


# RUN apt-get install python3.8 python3-pip -y

# RUN pip3 install tqdm

# RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html 