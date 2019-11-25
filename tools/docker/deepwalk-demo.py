# -*- coding: utf-8 -*-

import tensorflow as tf
import tf_euler


class DeepWalk(tf_euler.layers.Layer):
  def __init__(self, node_type, edge_type, max_id, dim,
               num_negs=8, walk_len=3, left_win_size=1, right_win_size=1):
    super(DeepWalk, self).__init__()
    self.node_type = node_type
    self.edge_type = edge_type
    self.max_id = max_id
    self.num_negs = num_negs
    self.walk_len = walk_len
    self.left_win_size = left_win_size
    self.right_win_size = right_win_size

    self.target_encoder = tf_euler.layers.Embedding(max_id + 1, dim)
    self.context_encoder = tf_euler.layers.Embedding(max_id + 1, dim)

  def call(self, inputs):
    src, pos, negs = self.sampler(inputs)
    embedding = self.target_encoder(src)
    embedding_pos = self.context_encoder(pos)
    embedding_negs = self.context_encoder(negs)
    loss, mrr = self.decoder(embedding, embedding_pos, embedding_negs)
    embedding = self.target_encoder(inputs)
    return (embedding, loss, 'mrr', mrr)

  def sampler(self, inputs):
    batch_size = tf.size(inputs)
    path = tf_euler.random_walk(
        inputs, [self.edge_type] * self.walk_len,
        default_node=self.max_id + 1)
    pair = tf_euler.gen_pair(path, self.left_win_size, self.right_win_size)
    num_pairs = pair.shape[1]
    src, pos = tf.split(pair, [1, 1], axis=-1)
    negs = tf_euler.sample_node(batch_size * num_pairs * self.num_negs,
                                self.node_type)
    src = tf.reshape(src, [batch_size * num_pairs, 1])
    pos = tf.reshape(pos, [batch_size * num_pairs, 1])
    negs = tf.reshape(negs, [batch_size * num_pairs, self.num_negs])
    return src, pos, negs

  def decoder(self, embedding, embedding_pos, embedding_negs):
    logits = tf.matmul(embedding, embedding_pos, transpose_b=True)
    neg_logits = tf.matmul(embedding, embedding_negs, transpose_b=True)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(logits), logits=logits)
    negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(neg_logits), logits=neg_logits)
    loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
    mrr = tf_euler.metrics.mrr_score(logits, neg_logits)
    return loss, mrr


if __name__ == '__main__':
    print("begin....")
    tf_euler.initialize_embedded_graph('ppi') # 图数据目录
    source = tf_euler.sample_node(128, tf_euler.ALL_NODE_TYPE)
    source.set_shape([128])

    model = DeepWalk(tf_euler.ALL_NODE_TYPE, [0, 1], 56944, 256)
    _, loss, metric_name, metric = model(source)

    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss, global_step)

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.train.MonitoredTrainingSession(
        hooks=[
            tf.train.LoggingTensorHook({'step': global_step,
                                  'loss': loss, metric_name: metric}, 100),
            tf.train.StopAtStepHook(2000)
        ]) as sess:
        while not sess.should_stop():
            sess.run(train_op)


