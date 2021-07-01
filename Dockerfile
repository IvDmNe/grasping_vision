FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

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
    gcc \
    cmake-curses-gui \
    build-essential \
    python3.8 \
    && rm -rf /var/lib/apt/lists/*

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ROS install
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update && apt-get install -y ros-melodic-robot ros-melodic-rosconsole
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc

# # Anaconda installing
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     bash Miniconda3-latest-Linux-x86_64.sh -b && \
#     rm Miniconda3-latest-Linux-x86_64.sh

# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"

# RUN conda init


RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
# RUN ln -sv /usr/bin/python3 /usr/bin/python

# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# RUN python3 -m pip install --upgrade pip==3.8.5

RUN pip3 install tensorboard cmake   # cmake from apt-get is too old
RUN pip3 install Pillow==8.2.0 numpy==1.18.5 torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html

RUN pip install 'git+https://github.com/facebookresearch/fvcore' opencv-python==4.5.2.54
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip3 install -e detectron2_repo

RUN pip3 install pandas rospkg scipy







# install cv_bridge for python3

RUN apt-get update && apt-get install -y python-catkin-tools python-dev libopencv-dev
RUN mkdir -p /cv_bridge_ws/src && \
    cd /cv_bridge_ws/src && \
    git clone https://github.com/kirillin/vision_opencv.git && \
    cd /cv_bridge_ws && \
    catkin config \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
        -DCMAKE_BUILD_TYPE=Release \
        -DSETUPTOOLS_DEB_LAYOUT=OFF \
        -Drosconsole_DIR=/opt/ros/melodic/share/rosconsole/cmake \
        -Drostime_DIR=/opt/ros/melodic/share/rostime/cmake \
        -Droscpp_traits_DIR=/opt/ros/melodic/share/roscpp_traits/cmake \
        -Dstd_msgs_DIR=/opt/ros/melodic/share/std_msgs/cmake \
        -Droscpp_serialization_DIR=/opt/ros/melodic/share/roscpp_serialization/cmake \
        -Dmessage_runtime_DIR=/opt/ros/melodic/share/message_runtime/cmake \
        -Dgeometry_msgs_DIR=/opt/ros/melodic/share/geometry_msgs/cmake \
        -Dsensor_msgs_DIR=/opt/ros/melodic/share/sensor_msgs/cmake \
        -Dcpp_common_DIR=/opt/ros/melodic/share/cpp_common/cmake && \
    cd src && git clone https://github.com/ros/catkin.git &&  cd .. && \
    catkin config --install && \
    catkin build cv_bridge && \
    echo "source /cv_bridge_ws/install/setup.bash --extend" >> ~/.bashrc




# install PCL library
RUN sudo apt-get install libpcl-dev -y

# clone project and move to it's dir
EXPOSE 11311
RUN git clone https://github.com/IvDmNe/point_cloud_processing.git
WORKDIR /point_cloud_processing/scripts

# download segmentation model weights
RUN wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
