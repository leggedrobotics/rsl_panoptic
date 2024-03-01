#=============================================================================
# Copyright (C) 2021, Robotic Systems Lab, ETH Zurich
# All rights reserved.
# http://www.rsl.ethz.ch
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# Authors: Julian Nubert, nubertj@ethz.ch
#          Pascal Roth, proth@ethz.ch
#          Lorenzo Terenzi, lterenzi@ethz.ch
#=============================================================================

#==
# Foundation
#==
ARG UBUNTU_VERSION=20.04
ARG CUDA=11.1.1
ARG DRIVER=510
ARG ARCH
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-cudnn8-devel-ubuntu${UBUNTU_VERSION} as base

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,video,compute,utility

# Suppresses interactive calls to APT
ENV DEBIAN_FRONTEND="noninteractive"

# Install graphics drivers
RUN apt update && apt install -y libnvidia-gl-${DRIVER} \
  && rm -rf /var/lib/apt/lists/*

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
ENV TERM=xterm-256color

# ----------------------------------------------------------------------------

#==
# System APT base dependencies and utilities
#==
RUN apt update && apt install -y \
  sudo \
  lsb-release \
  ca-certificates \
  apt-utils \
  gnupg2 \
  locate \
  curl \
  wget \
  git \
  vim \
  gedit \
  tmux \
  unzip \
  iputils-ping \
  net-tools \
  htop \
  iotop \
  iftop \
  nmap \
  software-properties-common \
  build-essential \
  gdb \
  pkg-config \
  cmake \
  zsh \
  tzdata \
  clang-format \
  clang-tidy \
  xterm \
  gnome-terminal \
  dialog \
  tasksel \
  && rm -rf /var/lib/apt/lists/*

#==
# ROS
#==
# Version
ARG ROS=noetic

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
 && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - \
 && apt update && apt install -y \
  python3-pip \
  python3-rosdep \
  python3-rosclean \
  python3-rosparam \
  python3-progressbar \
  python3-catkin-tools \
  python3-osrf-pycommon \
  python3-virtualenvwrapper \
  ros-${ROS}-desktop-full \
  ros-${ROS}-velodyne-pointcloud \
  ros-${ROS}-joy \
  ros-${ROS}-grid-map-core \
 && rm -rf /var/lib/apt/lists/* \
 && rosdep init && rosdep update \
 && apt update && apt install -y \
  libblas-dev \
  xutils-dev \
  gfortran \
  libf2c2-dev \
  libgmock-dev \
  libgoogle-glog-dev \
  libboost-all-dev \
  libeigen3-dev \
  libglpk-dev \
  liburdfdom-dev \
  liboctomap-dev \
  libassimp-dev \
  python3-catkin-tools \
  ros-${ROS}-ompl \
  ros-${ROS}-octomap-msgs \
  ros-${ROS}-pybind11-catkin \
  doxygen-latex \
  usbutils \
  python3-vcstool \
 && rm -rf /var/lib/apt/lists/* \
 && sudo ln -s /usr/include/eigen3 /usr/local/include/

#==
# RSL Panoptic Segmentation Setup
#==
COPY ./ /home/rsl_panoptic_seg

# Set CUDA paths
ENV CUDA_HOME=/usr/local/cuda-11.1 \
    PATH="/usr/local/cuda-11.1/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:${LD_LIBRARY_PATH}"

# Create a symbolic link for CUDA (do this before any CUDA detection happens)
RUN ln -s /usr/local/cuda-11.1 /usr/local/cuda

# Display CUDA environment variables (for debugging)
RUN echo "CUDA_HOME=$CUDA_HOME" && \
    echo "PATH=$PATH" && \
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Set the repository path
ENV REPO_PATH=/home/rsl_panoptic_seg

# Change to the repository directory
WORKDIR $REPO_PATH

# Install numpy and torch with specified versions
RUN pip3 install numpy==1.22.3
RUN pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install additional required packages
RUN pip3 install pandas tensorflow --upgrade wandb timm scikit-learn

# Install detectron2 if it's required for the project
RUN pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# Ensure pytube is installed before attempting to copy files
RUN pip3 install pytube

# Install the panoptic models package
RUN pip3 install -e ./panoptic_models

# Ensure the detectron2 evaluation directory exists before copying files
RUN mkdir -p /usr/local/lib/python3.8/dist-packages/detectron2/evaluation && \
    cp ./panoptic_models/panoptic_models/mask2former/third_party/adapted_source_code/panoptic_evaluation.py /usr/local/lib/python3.8/dist-packages/detectron2/evaluation/

# Set permissions for the dist-packages directory
RUN chmod 777 '/usr/local/lib/python3.8/dist-packages/'

ENV FORCE_CUDA="1"
# Build and install custom ops for Mask2Former
WORKDIR $REPO_PATH/panoptic_models/panoptic_models/mask2former/third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
RUN python3 setup.py build install

# Return to the repository path
WORKDIR $REPO_PATH

# ----------------------------------------------------------------------------

#==
# Cleanup
#==
RUN apt update && apt upgrade -y

# ----------------------------------------------------------------------------

#==
# Execution
#==
COPY bin/entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD []

# EOF