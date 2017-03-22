# -*- coding=utf-8 -*-
"""生成url向量表示"""
import math
import time

import numpy as np
import tensorflow as tf

import os
from dataset import url_log

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_string("worker_grpc_url", None,
                    "Worker GRPC URL (e.g., grpc://1.2.3.4:2222, or "
                    "grpc://tf-worker0:2222)")

# Flags for defining the tf.train.Server
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_integer('max_steps', 100000, 'number of train steps')
flags.DEFINE_string('logdir', '/tmp/log/word_embedding', 'the directory of log file')
flags.DEFINE_float('learn_rates', 1.0, 'the learning rates for training step')
flags.DEFINE_integer('summary_time', 5, 'the time between two summaries(n second)')
batch_size = 128
embedding_size = 200  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 128  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.
np.random.seed(FLAGS.task_index)
dataset = url_log.UrlRoute('20160727.csv', 50000, recordlen_min=3,
                           rand_biase=np.random.randint(0, 1000000))
vacob, reverse_vacob = dataset.build_dataset()
vacob_len = len(vacob)


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def train(server, cluster):

    num_workers = len(cluster.as_dict()['worker'])
    print 'num workers:%d' % num_workers

    if FLAGS.replicas_to_aggregate is None:
        replicas_to_aggregate = num_workers
    else:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate

    ischief = (FLAGS.task_index == 0)
    if ischief:
        tf.reset_default_graph()
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        with tf.name_scope('global_step'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
        with tf.name_scope('embedding_vector'):
            embedding = tf.Variable(tf.random_uniform([vacob_len, embedding_size], -1, 1))
        # weight and baise
        with tf.name_scope('nce_weight'):
            nce_weight = tf.Variable(
                tf.truncated_normal([vacob_len, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))

        with tf.name_scope('nce_biase'):
            nce_biase = tf.Variable(tf.zeros([vacob_len]))

        with tf.name_scope('imput'):
            train_input = tf.placeholder(tf.int32, shape=(batch_size))
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        embed = tf.nn.embedding_lookup(embedding, train_input)
        # loss
        with tf.name_scope('nce_loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weight, nce_biase, embed, train_labels,
                               num_sampled, vacob_len))
        with tf.name_scope('summaries'):
            variable_summaries(nce_biase, 'nce_biase')
            variable_summaries(nce_weight, 'nce_weight')
            variable_summaries(embedding, 'embedding')
            tf.scalar_summary('nec_loss', loss)
        summary_op = tf.merge_all_summaries()
        # Construct the SGD optimizer using a learning rate of 1.0.
        opt = tf.train.GradientDescentOptimizer(FLAGS.learn_rates)
        # sync the replicas
        opt = tf.train.SyncReplicasOptimizer(
            opt=opt,
            replicas_to_aggregate=replicas_to_aggregate,
            replica_id=FLAGS.task_index,
            total_num_replicas=num_workers,
            # use_locking=True,
            name='embedding_sync_replicas'
        )
        train_step = opt.minimize(loss, global_step=global_step)

        if ischief:
            # Initial token and chief queue runners required by the sync_replicas mode
            chief_queue_runner = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op()

        # Compute the similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embeddings = embedding / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sv = tf.train.Supervisor(
            is_chief=ischief,
            init_op=init,
            # logdir=FLAGS.logdir,
            # summary_op=tf.merge_all_summaries(),
            # saver=saver,
            global_step=global_step
        )

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            device_filters=['/job:ps', '/job:worker/task:%d' % FLAGS.task_index]
        )

        if ischief:
            print 'worker %d:starting initializing the session' % FLAGS.task_index
        else:
            print 'worker %d:waiting for session to be initialized' % FLAGS.task_index

        with sv.prepare_or_wait_for_session(server.target,
                                            config=sess_config) as sess:
            # queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            # sv.start_queue_runners(sess, queue_runners)
            # tf.logging.info('Started %d queues for processing input data.',
            #                 len(queue_runners))
            if ischief:
                # Chief worker(the master) will start the chief queue runner and call the init op
                sv.start_queue_runners(sess, [chief_queue_runner])
                assert init_tokens_op, 'empty init_token_ops'
                sess.run(init_tokens_op)

            local_step, time_begin = 0, time.time()
            summarytime = time_begin + FLAGS.summary_time

            writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)
            print 'train begins @ %f' % time_begin

            average_loss, start = 0, time.time()
            while True and not sv.should_stop():
                # print 'step to generate_batch'
                batch_inputs, batch_labels = dataset.generate_batch(
                    batch_size, num_skips, skip_window)
                feed_dict = {train_input: batch_inputs, train_labels: batch_labels}
                _, loss_val, step = sess.run([train_step, loss, global_step],
                                             feed_dict=feed_dict)
                print 'finish sess.run, step:%d' % step
                average_loss += loss_val
                local_step += 1
                if local_step % 2000 == 0 and local_step > 0:
                    now = time.time()
                    average_loss /= 2000
                    diff = (now - start) / 2000
                    print 'average loss at {} steps: {}\t in {} second'.format(
                        local_step, average_loss, diff)
                    average_loss = 0
                    start = now

                if local_step % 10000 == 0:
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

                if ischief and time.time() > summarytime:
                    summaries = sess.run(summary_op, feed_dict=feed_dict)
                    # sv.summary_computed(sess, summaries)
                    writer.add_summary(summaries, step)
                    summarytime += FLAGS.summary_time

                if step > FLAGS.max_steps:
                    break
            sv.stop()

            # final_embeddings = normalized_embeddings.eval()
            time_end = time.time()
            print("Training ends @ %f" % time_end)
            training_time = time_end - time_begin
            print("Training elapsed time: %f s" % training_time)
            if ischief:
                saver.save(sess, os.path.join(FLAGS.logdir, 'model.pkl'),
                           global_step=global_step)

def main(_):
    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.gfile.MakeDirs(FLAGS.logdir)
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    num_workers = len(worker_hosts)
    cluster = tf.train.ClusterSpec(
        cluster={'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        train(server, cluster)

if __name__ == '__main__':
    tf.app.run()


