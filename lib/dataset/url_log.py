# -*- coding=utf-8 -*-
# from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import cPickle as pickle
import collections
import os
import sys
import math
import re
import time
import codecs
import numpy as np
import pandas as pd
from multiprocessing import Process, Manager, Lock, Semaphore
import gc
import objgraph
# from six.moves import xrange

sys.path.append('..')
import dataset
import urlvacob
from dataset import url_category_sen
urllabel= url_category_sen.URLLABEL()

# filepath = os.path.join(os.path.dirname(__file__), '../..', 'data', 'log_data')
# filelist = os.listdir(filepath)
pd.set_option('display.encoding', 'UTF-8')

class UrlRoute(urlvacob.UrlVacob):

    def __init__(self, name, urlnum_max, data_path=None, recordlen_min=3, rand_biase=0):
        """
        :param name: string, the data file name
        :param urlnum_max: int, the max numbers of url
        :param data_path: the directory of the data. data_path
        :param recordlen_min: the minimal length of the url routes
        :param rand_biase: a random value to build a different dataset between different server
        """
        self.manager = Manager()
        self.lock = Lock()
        self.semaphore = Semaphore(4)
        urlvacob.UrlVacob.__init__(self, '', urlnum_max, 2)
        self._name = name
        self._data_path = self._get_default_path() if data_path == None \
            else data_path
        self._cache_path = os.path.join(dataset.ROOT_DIR, 'data', 'cache')
        self._rcdlen_min = recordlen_min
        self._window = 3 #default
        self._data_index = rand_biase
        self._routes = list() #process shared list


    def _load_excel(self, path, sheetname=None):
        """ """
        assert os.path.exists(path), \
            'path does not exist: {}'.format(path)
        return pd.read_excel(path, sheetname)


    def _sort_and_build_vacob_from_chunk(self, chunk):
        """ add the url to vacob and sort the chunk by ['IPTONUMBER', 'TIME']
            :param chunk: Dataframe, mainly use the coloumn
            :return the sorted chunk
        """
        self.create_url_vacob_from_list(chunk['request_url'.lower()].tolist())
        referer_url = chunk[chunk['referer_url'.lower()] != '-1']['referer_url'.lower()]
        self.create_url_vacob_from_list(referer_url.tolist())



    def getTermFromUrl(self, url):
        # Unify all external Search engines to External.Search.Engines
        InquirySuccess = r'^www\.made-in-china\.com/sendInquiry/success'
        InquiryFailure = r'^www\.made-in-china\.com/sendInquiry/failure'
        InquiryProd_ = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*made-in-china\.com/(sendInquiry|sendinquiry)/prod_'

        # Multi-search Pages Reg
        multisearch = r'^www\.made-in-china\.com/multi-search/.*/F[0-2]/'
        # multisearchCatalog = r'^www\.made-in-china.com/[^/]*-Catalog/'
        t = self.test_search(url)
        if t is not None:
            term = t

        # Unify all sendInquiry/success pages  to sendInquiry/success
        elif re.search(InquirySuccess, url) is not None:
            term = r'www.made-in-china.com/sendInquiry/success'

        # Unify all sendInquiry/failure pages to sendInquiry/failure
        elif re.search(InquiryFailure, url) is not None:
            term = r'www.made-in-china.com/sendInquiry/failure'

        # Unify all senInquiry/prods_ pages tp sendInqiry/prod_
        elif re.search(InquiryProd_, url) is not None:
            term = r'www.made-in-china\.com/sendInquiry/prods_item.html'

        # Unify all multi-search/.*F1 pages to www\.made-in-china\.multi-search/item/F1/pages\.html
        elif re.search(multisearch, url) is not None:
            term = r'www.made-in-china.com/multi-search/items/F1/pages.html'

        # # Unify all www\.made-in-china.com/multi-search/.*-Catalog/F0 pages to www.made-in-china.com/multi-search/items-Catalog/F0.html
        # elif re.search(multisearchCatalog, url) is not None:
        #     term = r'www.made-in-china.com/multi-search/items-Catalog/F0/pages.html'
        else:
            term = url
        return term

    def get_route_from_chunk(self, chunk, loaded, index, vacob_dic=None):
        start = time.time()
        row_iter = chunk.itertuples()
        last_ip = 0
        # path_list = list()
        path, tail = list(), None
        def update_path(path):
            new = []
            for url in path:
                if url in vacob_dic:
                    new.append(vacob_dic[url])
                else:
                    new.append(0)
            return new
        count = 0
        while True:
            try:
                # row[1]:time, row[2]:IP, row[3]:Request, row[4]:referer
                row = next(row_iter)
                if last_ip != row[2]:
                    if len(path) >= self._rcdlen_min:
                        path = update_path(path) if loaded else path
                        map(lambda x: self._routes.append(x), [-1] + path)
                    path = []
                    requseturl = self.getTermFromUrl(row[3])
                    referurl = self.getTermFromUrl(row[4])

                    if referurl != '-1' and re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', referurl) is None:
                        path = [referurl]
                    if re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', requseturl) is None:
                        path.append(requseturl)
                    if len(path) > 0:
                        last_ip = row[2]
                        tail = path[len(path) - 1]
                else:
                    requseturl = self.getTermFromUrl(row[3])
                    referurl = self.getTermFromUrl(row[4])
                    if referurl != '-1' and (referurl != tail or len(path) == 0)\
                            and (re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', referurl) is None):
                        path += [referurl]
                    if re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', requseturl) is None:
                        path += [requseturl]
                        tail = requseturl
                    elif len(path) > 0:
                        tail = path[len(path) - 1]
            except StopIteration:
                if len(path) >= self._rcdlen_min:
                    path = update_path(path) if loaded else path
                    map(lambda x: self._routes.append(x), [-1] + path)
                break
        diff = time.time() - start
        m, s = divmod(diff, 60)
        print "chunk_%d , spending time %d min %d s" % (index, m, s)

    def _get_route_from_chunk(self, d, lock, s, index, chunk, loaded=False, vacob_dic=None):
        """
        get the valid routes from chunk
        :param chunk: DataFrame, with four col ['TIME',
                'IPTONUMBER', 'REQUEST_URL', 'REFERER_URL']
        :param loaded: boolean, if true the vacob has been already build
        :param vacob_dic: dict, {'url':index}
        :return:
        """

        with s:

            print "process %d run for computing routes" % index
            # time.sleep(120)
            start = time.time()
            ip_series = chunk['iptonumber']
            # print chunk
            last_ip = ip_series[0]
            # path_list = list()
            path, tail = list(), None
            def update_path(path):
                new = []
                for url in path:
                    if url in vacob_dic:
                        new.append(vacob_dic[url])
                    else:
                        new.append(0)
                return new
            for i in xrange(len(ip_series)):
                ip, refer, request = chunk['iptonumber'][i], chunk['referer_url'][i], chunk['request_url'][i]
                if last_ip != ip:
                    with lock:
                        if len(path) >= self._rcdlen_min:
                            path = update_path(path) if loaded else path
                            map(lambda x: d.append(x), [-1] + path)
                    path = []
                    requseturl = self.getTermFromUrl(request)
                    referurl = self.getTermFromUrl(refer)

                    if referurl != '-1' and re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', referurl) is None:
                        path = [referurl]
                    if re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', requseturl) is None:
                        path.append(requseturl)
                    if len(path) > 0:
                        last_ip = ip
                        tail = path[len(path) - 1]
                else:
                    requseturl = self.getTermFromUrl(request)
                    referurl = self.getTermFromUrl(refer)
                    if referurl != '-1' and (referurl != tail or len(path) == 0) \
                            and (re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', referurl) is None):
                        path += [referurl]
                    if re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', requseturl) is None:
                        path += [requseturl]
                        tail = requseturl
                    elif len(path) > 0:
                        tail = path[len(path) - 1]
            with lock:
                if len(path) >= self._rcdlen_min:
                    path = update_path(path) if loaded else path
                    map(lambda x: d.append(x), [-1] + path)

            diff = time.time() - start
            m, s = divmod(diff, 60)
            print "process_%d exit, spending time %d min %d s" % (index, m, s)
        # print "process %d in and wait for computing routes" % index
        # with s:
        #
        #     print "process %d run for computing routes" % index
        #     # time.sleep(120)
        #     start = time.time()
        #     row_iter = chunk.itertuples()
        #     ip_series = chunk['iptonumber'.lower()].tolist()
        #     last_ip = ip_series[0]
        #     # path_list = list()
        #     path, tail = list(), None
        #     def update_path(path):
        #         new = []
        #         for url in path:
        #             if url in vacob_dic:
        #                 new.append(vacob_dic[url])
        #             else:
        #                 new.append(0)
        #         return new
        #     while True:
        #         try:
        #             # row[1]:time, row[2]:IP, row[3]:Request, row[4]:referer
        #             row = next(row_iter)
        #             if last_ip != row[2]:
        #                 with lock:
        #                     if len(path) >= self._rcdlen_min:
        #                         path = update_path(path) if loaded else path
        #                         map(lambda x: d.append(x), [-1] + path)
        #                 path = []
        #                 requseturl = self.getTermFromUrl(row[3])
        #                 referurl = self.getTermFromUrl(row[4])
        #
        #                 if referurl != '-1' and re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', referurl) is None:
        #                     path = [referurl]
        #                 if re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', requseturl) is None:
        #                     path.append(requseturl)
        #                 if len(path) > 0:
        #                     last_ip = row[2]
        #                     tail = path[len(path) - 1]
        #             else:
        #                 requseturl = self.getTermFromUrl(row[3])
        #                 referurl = self.getTermFromUrl(row[4])
        #                 if referurl != '-1' and (referurl != tail or len(path) == 0)\
        #                         and (re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', referurl) is None):
        #                     path += [referurl]
        #                 if re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', requseturl) is None:
        #                     path += [requseturl]
        #                     tail = requseturl
        #                 elif len(path) > 0:
        #                     tail = path[len(path) - 1]
        #             # if last_ip != row[2]:
        #             #     # create a new empty pathlist for the new ip and update the self._routes
        #             #     with lock:
        #             #         for path in path_list:
        #             #             if len(path) < self._rcdlen_min:
        #             #                 continue
        #             #             if loaded:
        #             #                 path = update_path(path)
        #             #             # self._routes.append(-1)
        #             #             map(lambda x: d.append(x), [-1] + path)
        #             #
        #             #     new_path = [row[4], row[3]] if row[4] != "-1" else [row[3]]
        #             #     path_list = [new_path]
        #             #     last_ip = row[2]
        #             #
        #             # else:
        #             #     change = False
        #             #     # search the path_list to find a path to append with url
        #             #     len_list = len(path_list)
        #             #     for i in xrange(len_list):
        #             #         # begin at the last path and  search forword
        #             #         path = path_list[len_list - i - 1]
        #             #         len_path = len(path)
        #             #         for j in xrange(len_path):
        #             #             if row[4] == path[len_path - j - 1]:
        #             #                 # find the proper path and the index
        #             #                 # create a new path by combining the slice of path and new url
        #             #                 if j == 0:
        #             #                     path.append(row[3])
        #             #                 else:
        #             #                     new_path = path[0: len_path - j] + [row[3]]
        #             #                     path_list.append(new_path)
        #             #                 change = True
        #             #                 break
        #             #         if change:
        #             #             break
        #             #     if not change:
        #             #         # do not search a proper path in the exist path_list
        #             #         # using new url create the new path
        #             #         new_path = [row[4], row[3]] if row[4] != "-1" else [row[3]]
        #             #         path_list.append(new_path)
        #         except StopIteration:
        #             with lock:
        #                 if len(path) >= self._rcdlen_min:
        #                     path = update_path(path) if loaded else path
        #                     map(lambda x: d.append(x), [-1] + path)
        #             # with lock:
        #             #     for path in path_list:
        #             #         if len(path) < self._rcdlen_min:
        #             #             continue
        #             #         if loaded:
        #             #             path = update_path(path)
        #             #         map(lambda x: d.append(x), [-1] + path)
        #             break
        #     diff = time.time() - start
        #     m, s = divmod(diff, 60)
        #     print "process_%d exit, spending time %d min %d s" % (index, m, s)

    def build_dataset(self):

        cache = os.path.join(self._cache_path, self._name + '.dat')
        loaded = False
        vacob_dic, reverse_vacob = None, None
        if os.path.exists(cache):
            with open(cache, 'rb') as file_obj:
                db = pickle.load(file_obj)
                self.set_vacob(db['vacob'])
                self.set_epoch_count(db['epoch_count'])
                # generate the dict vacob{'url': index} and reverse_vacob{index: 'url'}
                words = []
                for i in xrange(len(self.vacob)):
                    words.append(self.vacob[i][0])
                reverse_vacob = dict(zip(xrange(len(self.vacob)), words))
                vacob_dic = dict((zip(words, xrange(len(self.vacob)))))
                del words
                loaded = True
            print '{} loaded from cache {}'.format(self._name, cache)

            # return db['vacob'], db['routes'], db['reverse_vacob']


        file_path = os.path.join(self._data_path, self._name)
        assert os.path.exists(file_path), \
            'path does not exist: {}'.format(file_path)
        # 载入以csv格式保存的文件
        reader = pd.read_csv(file_path, sep=',', header=0,
                             #parse_dates={'TIME': [2, 3]}, 
                             chunksize=100000, encoding='utf-8')
        count = 0
        proc = list()
        for chunk in reader:
            # step 1: add the chunk's data to vacob, include the request url
            # and the referer url. the url must be non-null
            if not loaded:
                print "_sort_and_build_vacob_from_chunk_%d" % count
                self._sort_and_build_vacob_from_chunk(chunk)
            chunk = chunk.sort_values(['iptonumber'.lower(), 'visit_time'.lower()])
            # sel_col = {'iptonumber'.lower(): list(chunk['iptonumber'.lower()]),
            #            'request_url'.lower(): list(chunk['request_url'.lower()]),
            #            'referer_url'.lower(): list(chunk['referer_url'.lower()])
            #            }


            # step 2: aquire the route(represented as url) from the chunk
            self.get_route_from_chunk(
                chunk[['visit_time'.lower(), 'IPTONUMBER'.lower(), 'REQUEST_URL'.lower(), 'REFERER_URL'.lower()]],
                loaded, count,
                vacob_dic=vacob_dic)
            # p = Process(target=self._get_route_from_chunk,
            #             args=(self._routes, self.lock, self.semaphore, count,
            #                   sel_col,
            #                   loaded, vacob_dic))

            # proc.append(p)
            count += 1
        # for p in proc:
        #     p.start()
        # for p in proc:
        #     p.join()

        print len(self._routes)
        print len(self.vacob)

        #print '%s.%s(): %s' % (sys.getdefaultencoding.__module__, sys.getdefaultencoding.__name__, sys.getdefaultencoding())
        if not loaded:
            # sort the total vacob
            print "sorting the vacob begin"
            self.sort_vacob()
            print "sortint the vacob end"


            with codecs.open('route1.txt', 'w', encoding='utf-8') as file:
                i, last= 0, 0
                for num in xrange(len(self._routes)):
                    if self._routes[num] == -1:
                        line = str(i).decode('ascii')
                        s1 = str(',\t').decode('ascii').join(self._routes[last:num])
                        file.write(line + u',\t' + s1 + u'\n')
                        last = num + 1
                        i += 1
            # generate the dict vacob{'url': index} and reverse_vacob{index: 'url'}
            words = []
            for i in xrange(len(self.vacob)):
                words.append(self.vacob[i][0])
            reverse_vacob = dict(zip(xrange(len(self.vacob)), words))
            vacob_dic = dict(zip(words, xrange(len(self.vacob))))
            del words

            # step 3: after finishing building the vacob and url routes, represent
            # the routes by index
            routes = self._routes
            self._routes = []
            for url in routes:
                if url in vacob_dic:
                    self._routes.append(vacob_dic[url])
                else:
                    self._routes.append(-1 if url == -1 else 0)
            # step 4: dump the vacob and other data
            with open(cache, 'wb') as fi:
                db = {
                    'vacob': self.vacob,
                    # 'reverse_vacob': reverse_vacob
                    'epoch_count': self.epoch_count
                }
                pickle.dump(db, fi, pickle.HIGHEST_PROTOCOL)
            print 'file {} wrote to cache'.format(cache)
        # print self._routes[0:100]

        return self.vacob, reverse_vacob

    def get_reverse_vacob(self):
        cache = os.path.join(self._cache_path, self._name + '.dat')
        assert os.path.exists(cache), "the cache doesn't exist"
        with open(cache, 'rb') as file:
            db = pickle.load(cache)
            self.set_vacob(db['vacob'])
        words = []
        for vacob in self.vacob:
            words.append(vacob[0])
        reverse_vacob = dict(zip(xrange(self.vacob), words))
        return reverse_vacob



    def _get_default_path(self):
        """ """
        return os.path.join(dataset.ROOT_DIR, 'data', 'log_data')

    def extractUrl(dataframe):
        pass

    def generate_batch(self, batch_size, num_skips, skip_window, subsample=0):
        data_index = self._data_index % len(self._routes)
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)

        def subsampling(uid):
            # discard word with proability 1-(sqrt(subsample/freq)+(subsample/freq))
            ran = (math.sqrt(self.vacob[uid][1] / (subsample * self.epoch_count)) + 1) * (
                (subsample * self._epoch_count) / (self.vacob[uid][1]))
            return 0 if ran > np.random.random_sample() else -1

        def fill_url_queue(index):
            #填充连续的合法的url
            count, new_index = span, index
            while count > 0:
                if subsample > 0:
                    while 1:
                        if self._routes[new_index] == -1:
                            break
                        elif subsampling(self._routes[new_index]) == 0:
                            break
                        new_index = (new_index + 1) % len(self._routes)
                if self._routes[new_index] == -1:
                    count = span    #碰到分隔符，重新填充
                else:
                    buffer.append(self._routes[new_index])
                    count -= 1
                new_index = (new_index + 1) % len(self._routes)
            return new_index

        data_index = fill_url_queue(data_index)
        for i in range(batch_size // num_skips):
            target = skip_window #目标标签位置，对应于窗口的中心位置
            target_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in target_to_avoid:
                    target = np.random.randint(0, span)
                target_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j] = buffer[target]
            if subsample > 0:
                while 1:
                    if self._routes[data_index] == -1:
                        break
                    elif subsampling(self._routes[data_index]) == 0:
                        break
                        data_index = (data_index + 1) % len(self._routes)
            if self._routes[data_index] == -1:
                data_index = fill_url_queue(data_index)
                continue
            buffer.append(self._routes[data_index])
            data_index = (data_index + 1) % len(self._routes)
        self._data_index = data_index
        return batch, labels

    def distribution(self):
        cache = os.path.join(self._cache_path, self._name + '.dat')
        loaded = False
        vacob_dic, reverse_vacob = None, None
        if os.path.exists(cache):
            with open(cache, 'rb') as file_obj:
                db = pickle.load(file_obj)
                self.set_vacob(db['vacob'])
                # generate the dict vacob{'url': index} and reverse_vacob{index: 'url'}
                words = []
                for i in xrange(len(self.vacob)):
                    words.append(self.vacob[i])
            length = len(words)
            print "all words is {}".format(length)
            l = []
            l1 = []
            print 'the url and counts 0~10%:'
            for i in range(0, int(length * 0.1)):
                print words[i][0].encode('utf-8'), words[i][1]
                l[urllabel.get_sub_label_of_url(words[i][0].encode('utf-8'))] += 1
            print "tongjiqian10"
            for i in range(3000000):
                if l[i] > 0:
                    print "{} : {}".format(i, l[i])
            print "the url and counts 90%~end:"
            for i in range(int(length * 0.9), length):
                print words[i][0].encode('utf-8'), words[i][1]
                l1[urllabel.get_sub_label_of_url(words[i][0].encode('utf-8'))] += words[i][1]
            print "tongjihou10"
            for i in range(3000000):
                if l1[i] > 0:
                    print "{} : {}".format(i, l1[i])
                    #    reverse_vacob = dict(zip(xrange(len(self.vacob)), words))
                    #    vacob_dic = dict(zip(words, xrange(len(self.vacob))))
                    #    del words
                    #    loaded = True
                    # print '{} loaded from cache {}'.format(self._name, cache)
                    # print "total:",len(words)
                    # for i in range(len(words)):
                    # print words[i][0].encode('utf-8'),words[i][1]
                    # count = [0 for i in range(15)]
                    # for i in range(len(words)):
                    #     cc = words[i][1]
                    #     if cc < 2:
                    #         count[0]+=1
                    #     elif cc < 4:
                    #         count[1]+=1
                    #     elif cc<6:
                    #         count[2]+=1
                    #     elif cc<8:
                    #         count[3]+=1
                    #     elif cc < 10:
                    #         count[4]+=1
                    #     elif cc<15:
                    #         count[5]+=1
                    #     elif cc < 20:
                    #         count[6]+=1
                    #     elif cc<30:
                    #         count[7]+=1
                    #     elif cc<40:
                    #         count[8]+=1
                    #     elif cc<50:
                    #         count[9]+=1
                    #     elif cc<60:
                    #         count[10]+=1
                    #     elif cc<70:
                    #         count[11]+=1
                    #     elif cc<80:
                    #         count[12]+=1
                    #     elif cc<90:
                    #         count[13]+=1
                    #     else:
                    #         count[14]+=1
                    # self.plot(count,len(words))
                    # print "1:",l
                    # print "2~3(including 2 and 3):",p
                    # print "4~5:",m
                    # print "6~7:",n
                    # print "8~9:",a
                    # print "10~14:",b
                    # print "15~19:",o
                    # print "20~29:",c
                    # print "30~39:",d
                    # print "40~49:",e
                    # print "50~59:",f
                    # print "60~69:",g
                    # print "70~79:",h
                    # print "80~89:",j
                    # print "90~all:",k


        else:
            print 'No {} loaded form cache {}'.format(self._name, cache)

    # def plot(self, data, total):
    #     N = 15
    #     # menMeans = (20, 35, 30, 35, 27)
    #     # menStd =   (2, 3, 4, 1, 2)
    #
    #     ind = np.arange(N)  # the x locations for the groups
    #     width = 0.35  # the width of the bars
    #
    #     fig, ax = plt.subplots()
    #     rects1 = ax.bar(ind, data, width, color='r')
    #
    #     # womenMeans = (25, 32, 34, 20, 25)
    #     # womenStd =   (3, 5, 2, 3, 3)
    #     # rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)
    #
    #     # add some
    #     # ax.set_ylabel('Scores')
    #     ax.set_title('all words:{}'.format(total))
    #     ax.set_xticks(ind)
    #     ax.set_xticklabels(('1', '2~3', '4~5', '6~7', '8~9', '10~14', '15~19', '20~29',
    #                         '30~39', '40~49', '50~59', '60~69', '70~79', '80~89', '90~all'))
    #
    #     # ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )
    #     def autolabel(rects):
    #         # attach some text labels
    #         for rect in rects:
    #             height = rect.get_height()
    #             ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%d' % int(height),
    #                     ha='center', va='bottom')
    #
    #     autolabel(rects1)
    #     # autolabel(rects2)
    #     plt.show()


if __name__ == '__main__':
    import time
    start = time.time()
    urldb = UrlRoute('20160720.csv', 300000, recordlen_min=3)
    vocab, reverse_vacob = urldb.build_dataset()
    lenght = len(vocab)
    # for i in xrange(100):
    #     print "%d: %s" % (vocab[lenght - i - 1][1], vocab[lenght - i -1])
    # for i in xrange(100):
    #     print "%d: %s" % (vocab[i][1], vocab[i])
    diff = time.time() - start
    print "build dataset uses time:%d min %d second" % (diff//60, diff - int(diff//60) * 60)
    print urldb.generate_batch(batch_size=128, num_skips=2, skip_window=1)
