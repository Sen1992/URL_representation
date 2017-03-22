import time
import math
import os
import cPickle as pickle
import tensorflow as tf
import numpy as np


embedding_size = 300  # Dimension of the embedding vector.
valid_window = 2000
valid_size = 64
valid_examples = np.random.choice(range(100, 500), valid_size, replace=False)
out_dir = os.path.join(os.path.dirname(__file__), '../..', 'out')
def load_vacob(path):
    assert os.path.exists(path), "the file doesn't exist %s" % path
    with open(path, 'rb') as file:
        db = pickle.load(file)
        words = []
        for url in db['vacob']:
            words.append(url[0])
        return dict(zip(xrange(len(words)), words)), len(words)
def show_nearest():
    pass
def build_eva():
    # with tf.Graph.as_default() as graph:
    path = os.path.join(os.path.dirname(__file__), '../..', 'data', 'cache', '20160720.csv.dat')
    reverse_vacob, vacob_len = load_vacob(path)
    for i in xrange(100):
        print "%d:\t%s" % (i, reverse_vacob[i])

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.name_scope('embedding_vector'):
        embedding = tf.Variable(tf.random_uniform([vacob_len, embedding_size], -1, 1))
        #embed = tf.nn.embedding_lookup(embedding, train_input)

    # weight and baise
    with tf.name_scope('nce_weight'):
        nce_weight = tf.Variable(
            tf.truncated_normal([vacob_len, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))

    with tf.name_scope('nce_biase'):
        nce_biase = tf.Variable(tf.zeros([vacob_len]))

    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embeddings = embedding / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    # similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, out_dir + '/model.ckpt-160001')
        # sess.run(tf.initialize_all_variables())
        sim = similarity.eval()
        for i in range(valid_size):
            top_k = 8
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % reverse_vacob[valid_examples[i]]
            for k in range(top_k):
                log_str = "%s %s," % (log_str, reverse_vacob[nearest[k]])
            print log_str

if __name__ == "__main__":
    build_eva()
# with tf.name_scope('nce_loss'):
#     loss = tf.reduce_mean(
#         tf.nn.nce_loss(nce_weight, nce_biase, embed, train_labels,
#                        num_sampled, vacob_len))

