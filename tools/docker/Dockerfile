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

RUN wget https://mirrors.aliyun.com/apache/hadoop/common/hadoop-3.1.3/hadoop-3.1.3.tar.gz && \
    tar -zxf hadoop-3.1.3.tar.gz -C /usr/local  && \
    cd /usr/local/  && \
    mv ./hadoop-3.1.3/ ./hadoop
ENV HADOOP_HOME /usr/local/hadoop
ENV LIBRARY_PATH $LIBRARY_PATH:$HADOOP_HOME/lib/native
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$HADOOP_HOME/lib/native
ENV PATH $PATH:$HADOOP_HOME/bin
RUN sh -c 'echo export CLASSPATH=$CLASSPATH:$(hadoop classpath --glob) >> /etc/bash.bashrc'

RUN mkdir -p /root/.pip
COPY tools/docker/pip.conf /root/.pip
RUN pip --no-cache-dir install numpy==1.16.4 tensorflow==1.12.3 setuptools

RUN apt-get install unzip git curl vim -y
#  && cd /usr/local/lib/python2.7/dist-packages/tensorflow && rm -rf libtensorflow_framework.so && ln -s libtensorflow_framework.so.1 libtensorflow_framework.so

RUN mkdir /root/quick-begin && cd /root/quick-begin && \
    curl -k -O https://raw.githubusercontent.com/alibaba/euler/master/examples/ppi_data.py && \
	pip install networkx==1.11 scikit-learn==0.19.0 scipy==1.2.0

RUN apt-get update && apt-get install -y openjdk-8-jdk

RUN cd /root/ && wget https://mirrors.tuna.tsinghua.edu.cn/apache/zookeeper/zookeeper-3.5.5/apache-zookeeper-3.5.5-bin.tar.gz && \
    tar -zxf apache-zookeeper-3.5.5-bin.tar.gz
RUN cd /root/ && cp -rf apache-zookeeper-3.5.5-bin /root/server1 && \
	cp -rf apache-zookeeper-3.5.5-bin /root/server2 && \
	cp -rf apache-zookeeper-3.5.5-bin /root/server3 && \
	mkdir /root/server1/data && echo "1" > /root/server1/data/myid && \
	mkdir /root/server2/data && echo "2" > /root/server2/data/myid && \
	mkdir /root/server3/data && echo "3" > /root/server3/data/myid
COPY ./tools/docker/server1/zoo.cfg /root/server1/conf
COPY ./tools/docker/server2/zoo.cfg /root/server2/conf
COPY ./tools/docker/server3/zoo.cfg /root/server3/conf

RUN echo "export PATH=$PATH:/root/server1/bin/" >> /root/.bashrc
RUN echo "service ssh restart" >> /root/.bashrc

RUN apt-get install -y openssh-server && ssh-keygen -f /root/.ssh/id_rsa -t rsa -N '' && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

COPY . /tmp/Euler
RUN mv -f /tmp/Euler/gflags /tmp && cd /tmp/gflags && mv -f build2 build && cd build && \
    mkdir -p CMakeFiles/gflags_shared.dir/src/ && mkdir -p CMakeFiles/gflags_nothreads_shared.dir/src/ &&\
	mkdir lib && mkdir bin && \
	make && make install
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
	
RUN cd /root/quick-begin && python ppi_data.py

COPY ./tools/docker/data /tmp/data
COPY ./tools/docker/dist_tf_euler.sh /tmp/
COPY ./tools/docker/dist_start.sh /tmp/
COPY ./tools/docker/hdf-init.sh /tmp/
COPY ./tools/docker/hdfs-conf/capacity-scheduler.xml $HADOOP_HOME/etc/hadoop/
COPY ./tools/docker/hdfs-conf/core-site.xml $HADOOP_HOME/etc/hadoop/
COPY ./tools/docker/hdfs-conf/hadoop-env.sh $HADOOP_HOME/etc/hadoop/
COPY ./tools/docker/hdfs-conf/hdfs-site.xml $HADOOP_HOME/etc/hadoop/
COPY ./tools/docker/hdfs-conf/mapred-site.xml $HADOOP_HOME/etc/hadoop/
COPY ./tools/docker/hdfs-conf/yarn-site.xml $HADOOP_HOME/etc/hadoop/
COPY ./tools/docker/deepwalk-demo.py /root/quick-begin
COPY ./tools/docker/graphsage-demo.py /root/quick-begin

RUN apt-get install -y openssh-server 

RUN echo 'root:123456' |chpasswd  && \
	sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
	sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config && \
	echo 'export $(cat /proc/1/environ |tr "\\0" "\\n" | xargs)'　>> /etc/profile

EXPOSE 22
