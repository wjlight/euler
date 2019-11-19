#!/bin/bash

bash dist_tf_euler.sh \
	--data_dir hdfs://localhost:9000/euler/data \
        --euler_zk_addr 127.0.0.1:2181 --euler_zk_path /test/tf_euler \
        --max_id 56944 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 \
        --model graphsage_supervised --mode train
