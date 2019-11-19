#!/bin/bash


/root/server1/bin/zkServer.sh start
/root/server2/bin/zkServer.sh start
/root/server3/bin/zkServer.sh start

cd /usr/local/hadoop
./bin/hdfs namenode -format
./sbin/start-dfs.sh
./sbin/start-yarn.sh

jps

