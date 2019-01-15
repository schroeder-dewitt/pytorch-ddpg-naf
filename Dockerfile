FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
# FROM ubuntu:16.04
MAINTAINER Christian Schroeder de Witt

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*


#Install python3 pip3
RUN apt-get update
#RUN apt-get -y install python3
#RUN apt-get install -y python-apt --reinstall
RUN add-apt-repository ppa:jamesh/snap-support && apt-get update && apt install -y patchelf
#RUN add-apt-repository ppa:jonathonf/python-3.6 -y && apt-get update && apt-get install -y python3.6
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y python3.6 python3.6-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py
RUN pip3 install --upgrade pip
RUN apt-get install -y python-apt --reinstall
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger

RUN mkdir /install
WORKDIR /install

#### -------------------------------------------------------------------
#### install MongoDB (for Sacred)
#### -------------------------------------------------------------------

# Install pymongo
RUN pip3 install pymongo

#### -------------------------------------------------------------------
#### install pysc2 #(from Mika fork)
#### -------------------------------------------------------------------

# RUN git clone https://github.com/samvelyan/pysc2.git /install/pysc2
RUN git clone https://github.com/deepmind/pysc2.git /install/pysc2
RUN pip3 install /install/pysc2/

#RUN pip3 install /install/pysc2/

#### -------------------------------------------------------------------
#### install Sacred (from OxWhirl fork)
#### -------------------------------------------------------------------

RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install OpenBW and TorchCraft
#### -------------------------------------------------------------------

## --------- sdl
# RUN apt-get update && apt-get install -y libsdl2-dev libsdl2-2.0

## --------- build OpenBW
# RUN git clone https://github.com/openbw/openbw
# RUN git clone https://github.com/openbw/bwapi
# RUN cd bwapi && mkdir build && cd build \
#    && cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENBW_DIR=../../openbw -DOPENBW_ENABLE_UI=$OBW_GUI \
#    && make && make install

# -------- TorchCraft

# RUN git clone https://github.com/torchcraft/torchcraft.git -b develop --recursive

## --------- zmq
# RUN apt-get purge -y libzmq*
# RUN apt-get update && apt-get install -y libtool pkg-config build-essential autoconf automake uuid-dev
# RUN wget https://github.com/zeromq/libzmq/releases/download/v4.2.2/zeromq-4.2.2.tar.gz
# RUN tar xvzf zeromq-4.2.2.tar.gz
# RUN ulimit -n 1000 && apt-get update
# RUN cd zeromq-4.2.2 && ./configure && make install && ldconfig

## --------- zstd
# This can only be done on 18.04, so commenting out for now
# RUN apt-get install -y libzstd1-dev zstd
# RUN wget https://github.com/facebook/zstd/archive/v1.1.4.tar.gz
# RUN tar xf v1.1.4.tar.gz
# RUN cd zstd-1.1.4/ && make -j && make install && ldconfig

## --------- build BWEnv
# RUN cd torchcraft/BWEnv && mkdir build && cd build && sed -i.bak -e '5d;' ../CMakeLists.txt
# RUN cd torchcraft/BWEnv/build && cmake .. -DCMAKE_BUILD_TYPE=relwithdebinfo -DZMQ_LIBRARIES=/usr/local/lib/libzmq.so  \
#    && make -j
# RUN echo "cp /install/torchcraft/BWEnv/build/BWEnv.so /pymarl/3rdparty/StarCraftI/linux/bots/BWEnv.so" >> ~/.bashrc

## --------- python client deps
RUN pip3 install pybind11

## --------- python client
# RUN cd torchcraft && pip3 install .

#### -------------------------------------------------------------------
#### final steps
#### -------------------------------------------------------------------
RUN pip3 install pygame

#### -------------------------------------------------------------------
#### Plotting tools
#### -------------------------------------------------------------------

# RUN apt-get -y install ipython ipython-notebook
RUN pip3 install statsmodels pandas seaborn
RUN mkdir /pool && echo "export PATH=$PATH:'/pool/pool'" >> ~/.bashrc
# RUN cd /pool && git clone https://github.com/oxwhirl/pool.git pool-repo &&  ln -s pool-repo/pool && git submodule update --init --recursive

RUN pip3 install cloudpickle ruamel.yaml

RUN apt-get install -y libhdf5-serial-dev cmake
RUN git clone https://github.com/Blosc/c-blosc.git /install/c-blosc && cd /install/c-blosc && cmake -DCMAKE_INSTALL_PREFIX=/usr/local && cmake --build . --target install
RUN pip3 install tables h5py

#### -------------------------------------------------------------------
#### install tensorflow
#### -------------------------------------------------------------------
RUN pip3 install tensorflow-gpu tensorboardX tqdm
#RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp34-cp34m-linux_x86_64.whl

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------

RUN pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
RUN pip3 install torchvision snakeviz pytest probscale
RUN apt-get install -y htop iotop

#### -------------------------------------------------------------------
#### install mujoco
#### -------------------------------------------------------------------
RUN apt install -y libosmesa6-dev libglew1.5-dev
# RUN add-apt-repository ppa:jamesh/snap-support && apt-get update && apt install -y patchelf
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN export PATH=$PATH:$HOME/.local/bin
COPY ./mujoco_key.txt /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}

RUN pip3 install gym[mujoco] --upgrade
RUN pip3 install mujoco-py
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin" >> ~/.profile
RUN python3 -c "import mujoco_py"

WORKDIR /install
RUN git clone https://github.com/schroeder-dewitt/multiagent-particle-envs.git /install/multiagent && cd /install/multiagent && pip3 install -e .

RUN pip3 install blosc

# needed for matplab
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y tzdata
RUN apt-get install -y python3.6-tk

# set pythonpath
RUN echo "export PYTHONPATH=/pymarl/src" >> ~/.bashrc
RUN echo "export PYTHONPATH=/pymarl/src" >> ~/.profile

EXPOSE 8888

WORKDIR /naf_ikostrikov
# RUN echo "mongod --fork --logpath /var/log/mongod.log" >> ~/.bashrc
#CMD ["mongod", "--fork", "--logpath", "/var/log/mongod.log"]
# EXPOSE 27017
# EXPOSE 28017

# CMD service mongod start && tail -F /dev/null
