import time
import sys
import os
import cPickle as pickle
from multiprocessing import Pool, Manager

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np
sys.path.append("..")
from dataset import url_category_sen
from get_model_tensor import print_tensors_in_checkpoint_file
embedding_size = 300  # Dimension of the embedding vector.


out_dir = os.path.join(os.path.dirname(__file__), '../..', 'out')
path = os.path.join(os.path.dirname(__file__), '../..', 'data', 'cache', '20160720.csv.dat')
urllabel= url_category_sen.URLLABEL()


def load_vacob(path):
    assert os.path.exists(path), "the file doesn't exist %s" % path
    with open(path, 'rb') as file:
        db = pickle.load(file)
        words = []
        for url in db['vacob']:
            words.append(url[0])
        return dict(zip(xrange(len(words)), words)), len(words)


def load_embeding_by_type(valid_id, start_key, end_key, reverse_vacob, url_dic, lock):
    """
    :param typeId:
    :param reverse_vacob:
    :return:
    """
    # url_dic, index = {x: [] for x in valid_id}, 0
    for key in xrange(start_key, end_key):
        url_label = urllabel.get_sub_label_of_url(reverse_vacob[key])
        if url_label in list(valid_id):
            print "url_label %d" % url_label
            with lock:
                url_dic[url_label] += [key]
    print "exit"


def get_words_of_types(valid_ids=None):
    """

    :param valid_ids: list
    :return:
    """
    if valid_ids is None:
        valid_size = 80
        valid_sample = np.random.choice(range(0, urllabel.lenght), valid_size, replace=False)
        valid_ids = urllabel.getid(valid_sample)

    pool = Pool(8)
    manager = Manager()
    lock = manager.Lock()

    reverse_vacob, vacob_len = load_vacob(path)
    for i in xrange(100000000):
        count = i
    print "vacob length %d" % vacob_len
    vacob = manager.dict(reverse_vacob)
    del reverse_vacob
    valid_sets = manager.dict({x: [] for x in valid_ids})
    start = time.time()
    print "start loading  valid_sets"
    for i in range(16):
        step_size = int(vacob_len / 16)
        start_key, end_key = step_size * i, min(step_size * (i + 1), vacob_len)
        pool.apply_async(load_embeding_by_type, args=(valid_ids, start_key, end_key,
                                                      vacob, valid_sets, lock))
    pool.close()
    pool.join()
    print "load valid_sets end, spending %d min" % ((time.time() - start) / 60)
    return dict(valid_sets)


def calculate_variance_on_types(valid_sets):
    """

    :param valid_sets:dict {type_id: embs_index}
    :return:
    """
    embedding = np.array(print_tensors_in_checkpoint_file("embedding_vector/Variable"), dtype=np.float32)
    file = open('out_exp1.txt', 'a+')
    file.write("########################################################")
    for key in valid_sets.keys():
        if len(valid_sets[key]) == 0:
            continue
        log_str = "\ntype Id:%d\t" % key
        # print log_str
        file.write(log_str)
        valid_embedding = embedding[valid_sets[key]]
        mean = valid_embedding.mean(axis=0, dtype=np.float32)
        mean = mean.reshape(-1)
        # print "mean:\n" % mean
        variance = np.sum(np.square(valid_embedding - mean)) / (len(valid_sets[key])*embedding_size)
        sq = np.sqrt(variance)
        # print "variance:\n %f" % variance
        file.write("variance:%f\tstddev:%f\tnums:%d\n" % (variance, sq, len(valid_sets[key])))
    file.close()


def calculate_variance_on_all():
    embedding = np.array(print_tensors_in_checkpoint_file("embedding_vector/Variable"), dtype=np.float32)
    mean = embedding.mean(axis=0, dtype=np.float32)
    mean = mean.reshape(-1)
    variance = np.sum(np.square(embedding - mean)) / (len(embedding) * embedding_size)
    sq = np.sqrt(variance)
    print "variance:%f\tstddev:%f\t" % (variance, sq)


def show_word_by_id(typeid, num):
    reverse_vacob, vacob_len = load_vacob(path)
    fid = open('urls_by_type.txt', 'w')
    log_str = "##############################\ntypeId:%d" % typeid
    for key in reverse_vacob:
        if urllabel.get_sub_label_of_url(reverse_vacob[key]) == typeid:
            log_str = "%s %s," % (log_str, reverse_vacob[key])
            num -= 1
        if num == 0:
            break
    fid.write(log_str)


def plot_with_labels(low_dim_embs, labels, color, scalarmap, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches

    for i, label in zip(range(len(labels)), labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, c=scalarmap.to_rgba(color[i]))
        # plt.annotate(label,
        #              xy=(x, y),
        #              xytext=(5, 2),
        #              textcoords='offset points',
        #              ha='right',
        #              va='bottom')
    plt.savefig(filename)


def plot_with_labels_1(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches

    for i, label in zip(range(len(labels)), labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

def visual_embs_by_tsne(embeding, labels, color=None, scalarmap=None):
    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=6000)
    low_dims_emb = tsne.fit_transform(embeding)
    if color is not None:
        plot_with_labels(low_dims_emb, labels, color, scalarmap)
    else:
        plot_with_labels_1(low_dims_emb, labels, filename="tsne-500.1.png")

def visualization(valid_sets):

    # valid_sets = {x:[]}
    # db_index, labels, category = dict(), dict(), 0
    db_index, labels, category = list(), list(), 0
    color = []
    for key in valid_sets:
        length = len(valid_sets[key])
        if length > 100:
            if length > 1000:
                for j in list(np.random.choice(range(0, length), 1000, replace=False)):
                    db_index.append(valid_sets[key][j])
                labels += 1000 * [key]
                color += 1000 * [category]
                # db_index[category] = list(np.random.choice(valid_sets[key], 200))

            else:
                db_index += valid_sets[key]
                labels += length * [key]
                color += length * [category]
                # db_index[category] = valid_sets[key]
            # labels[category] = key
            category += 1
        if category >= 5:
            break
    cNorm = cmx.colors.Normalize(vmin=0, vmax=category)
    hot = plt.get_cmap('Set1')
    scarlarmap = cmx.ScalarMappable(cNorm, hot)
    embedding = np.array(print_tensors_in_checkpoint_file("embedding_vector/Variable"), dtype=np.float32)
    visual_embs_by_tsne(embedding[db_index], labels, color, scarlarmap)

def test():
    vacob = None
    import codecs
    with open(path, 'rb') as file:
        db = pickle.load(file)
        vacob = db['vacob']
    file = codecs.open("vacob_1.txt", 'w', encoding='utf-8')
    for word in vacob:
        # if re.match(r'(((&|\?)(key)?word=)|(^membercenter\.))', reverse_vacob[key]) is not None:
        file.write(u"%s: %d\n" % (word[0], word[1]))

if __name__ == "__main__":
    # build_eva()
    # calculate_variance_on_all()
    # show_word_by_id(7013, 8)
    # test()
    valid_sets = get_words_of_types([1165, 200, 7008, 2462])
    calculate_variance_on_types(valid_sets)
    calculate_variance_on_all()

    visualization(valid_sets)

    vacob, vacob_len = load_vacob(path)
    labels = []
    for i in range(1, 500):
        labels.append(urllabel.get_sub_label_of_url(vacob[i]))
    embedding = np.array(print_tensors_in_checkpoint_file("embedding_vector/Variable"),
                         dtype=np.float32)
    visual_embs_by_tsne(embedding[1:500, :], labels)




