# -*- coding=utf-8 -*-
"""生成url向量表示,路径中不包含卖家信息"""
import math
import time
import os.path as osp

import numpy as np
import tensorflow as tf

from dataset import url_log

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_steps', 160000, 'number of train steps')
flags.DEFINE_string('logdir', '/tmp/log/word_embedding', 'the directory of log file')
flags.DEFINE_float('learn_rates', 0.7, 'the learning rates for training step')
flags.DEFINE_integer('summary_time', 50, 'the time between two summaries(n second)')

out_dir = osp.join(osp.dirname(__file__), '..', 'out', 'sub_sample')
if not osp.exists(out_dir):
    tf.gfile.MakeDirs(out_dir)
batch_size = 128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 4         # How many times to reuse an input to generate a label.
subsample = 1e-3
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 128  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.


dataset = url_log.UrlRoute('20160720.csv', 300000, recordlen_min=3)
vacob, reverse_vacob = dataset.build_dataset()
vacob_len = len(vacob)
def train():
    graph = tf.Graph()
    with graph.as_default():
	global_step = tf.Variable(0, trainable= False)
        with tf.name_scope('imput'):
            train_input = tf.placeholder(tf.int32, shape=(batch_size,))
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        def variable_summaries(var, name):
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.scalar_summary('mean/' + name, mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('stddev/'+name, stddev)
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
                tf.histogram_summary(name, var)

        with tf.device('/cpu:0'):
            with tf.name_scope('embedding_vector'):
                # embedding = tf.Variable(tf.random_uniform([vacob_len, embedding_size],
                #                                           -0.5/embedding_size, -0.5/embedding_size))
                embedding = tf.Variable(tf.random_uniform([vacob_len, embedding_size],
                                                          -1, 1))
                embed = tf.nn.embedding_lookup(embedding, train_input)
                variable_summaries(embedding, 'embedding')
            #weight and baise
            with tf.name_scope('nce_weight'):
                nce_weight = tf.Variable(
                    tf.truncated_normal([vacob_len, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))
                variable_summaries(nce_weight, 'nce_weight')
            with tf.name_scope('nce_biase'):
                nce_biase = tf.Variable(tf.zeros([vacob_len]))
                variable_summaries(nce_biase, 'nce_biase')

            #loss
        with tf.name_scope('nce_loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weight, nce_biase, embed, train_labels,
                               num_sampled, vacob_len))
            tf.scalar_summary('nec_loss', loss)
        # Construct the SGD optimizer using a learning rate of 1.0.
        learning_rate = tf.train.exponential_decay(FLAGS.learn_rates, global_step, 20000, 0.9, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

        #Compute the similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embeddings = embedding / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(FLAGS.logdir, session.graph)
        average_loss, start, last_summary_time = 0, time.time(), 0
        # step = 0
        remaining = 0

        while True:
            batch_inputs, batch_labels = dataset.generate_batch(
                batch_size, num_skips, skip_window, subsample)
            feed_dict = {train_input: batch_inputs, train_labels: batch_labels}

            _, loss_val, rate, step = session.run([optimizer, loss, learning_rate, global_step], feed_dict=feed_dict)
            average_loss += loss_val
            now = time.time()

            if now - last_summary_time > FLAGS.summary_time:
                summaries = session.run(merged, feed_dict=feed_dict)
                writer.add_summary(summaries, step)
                last_summary_time = now

            if step % 2000 == 0 and step > 0:
                average_loss /= 2000
                diff = (time.time() - start) / 2000
                remaining = diff * (FLAGS.max_steps - step)
                m, s = divmod(remaining, 60)
                h, m = divmod(m, 60)
                print 'average loss at %d steps: %f\t in %f sec (lr:%f)' % (
                    step, average_loss, diff, rate)
                print 'remaining time: %d hour %d min %d sec' % (h, m, s)
                average_loss = 0
                start = time.time()


            if step % 30000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_vacob[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_vacob[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
            if step > FLAGS.max_steps:
                break
            #step += 1
        final_embeddings = normalized_embeddings.eval()
        saver.save(session, out_dir + '/model.ckpt', global_step=step)

    # with tf.Session(graph) as sess:
    #     saver.restore(sess, out_dir + '/model.ckpt-40001')
    #     # print sess.run(embedding)
    #     print embedding.eval()

if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.gfile.MakeDirs(FLAGS.logdir)
    train()
