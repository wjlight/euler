FROM ubuntu:18.04

COPY tools/docker/sources.list /etc/apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    ant \
    autoconf \
    build-essential \
    cmake \
    default-jre \
    golang-go \
    python-dev \
    python-pip \
    python-setuptools \
    wget
#   && \
#  apt-get clean && \
#  rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$JAVA_HOME/lib/server

RUN wget https://mirrors.aliyun.com/apache/hadoop/common/hadoop-2.9.2/hadoop-2.9.2.tar.gz && \
    tar xf hadoop-2.9.2.tar.gz -C /usr/local && \
    rm -rf hadoop-2.9.2.tar.gz
ENV HADOOP_HOME /usr/local/hadoop-2.9.2
ENV LIBRARY_PATH $LIBRARY_PATH:$HADOOP_HOME/lib/native
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$HADOOP_HOME/lib/native
ENV PATH $PATH:$HADOOP_HOME/bin
RUN sh -c 'echo export CLASSPATH=$CLASSPATH:$(hadoop classpath --glob) >> /etc/bash.bashrc'

RUN mkdir -p /root/.pip
COPY tools/docker/pip.conf /root/.pip
RUN pip --no-cache-dir install numpy==1.16.4 tensorflow==1.12.3 setuptools

RUN apt-get install unzip git curl vim -y
#  && cd /usr/local/lib/python2.7/dist-packages/tensorflow && rm -rf libtensorflow_framework.so && ln -s libtensorflow_framework.so.1 libtensorflow_framework.so

COPY . /tmp/Euler
RUN mv -f /tmp/Euler/gflags /tmp && cd /tmp/gflags && mv -f build2 build && cd build && \
    mkdir -p CMakeFiles/gflags_shared.dir/src/ && mkdir -p CMakeFiles/gflags_nothreads_shared.dir/src/ &&\
	mkdir lib && make && make install
RUN cd /tmp/Euler/third_party/zookeeper && \
    ((cd zookeeper-client/zookeeper-client-c; \
      [ -e generated/zookeeper.jute.h ] && [ -e generated/zookeeper.jute.c ]) || \
     ant compile_jute)

#in docker, must build glfags below:
#apt-get install cmake-curses-gui
#mkdir /tmp/gflags/build  && cd /tmp/gflags/build && ccmake .. && make && make install

RUN cd /tmp/Euler && \
    mkdir -p /tmp/build && cd /tmp/build && \
    cmake /tmp/Euler && \
    make -j $(expr $(nproc) \* 2) && \
    cd /tmp/Euler && \
    python tools/pip/setup.py install

RUN mkdir /root/quick-begin && cd /root/quick-begin && \
    curl -k -O https://raw.githubusercontent.com/alibaba/euler/master/examples/ppi_data.py && \
	pip install networkx==1.11 scikit-learn==0.19.0 scipy==1.2.0 && \
	python ppi_data.py