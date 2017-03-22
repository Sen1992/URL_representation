# -*- coding=utf-8 -*-
from __future__ import division

import os
import cPickle
import re

MAX_HASH_VACOB = 50000000    #最大的单词库数量，可以包括30*0.7=21M的url
MAX_LEN_URL = 4096  #url最大长度
INF = 35997760
class UrlVacob:
    """
    :argument

    """

    def __init__(self, vacob_path, maxUrlNum, min_count):
        self._vacob_path = vacob_path #词表路径
        assert maxUrlNum < MAX_HASH_VACOB * 0.7
        self._maxUrlNum = int(maxUrlNum*1.4) #词表大小
        self._vacob_hash = [-1 for _ in xrange(self._maxUrlNum)] #哈希编码
        self._count = 0 #当前词汇表大小
        self._vacob = [] #词汇表[['url', count]...['url', count]]
        self._min_count = min_count #url最小频率
        self._min_reduce = 1 #reduce操作的参数，初始值为2
        self._epoch_count = 0


    def geturlhash(self, url):
        """获得url的hash值"""
        hashcode = 1
        for i in xrange(len(url)):
            hashcode = hashcode*256 + ord(url[i])
        hashcode = hashcode % self._maxUrlNum
        return hashcode


    def addurltovacob(self, url):
        """添加新的url到词汇表
        return:
        """
        assert len(url) <= MAX_LEN_URL, 'url长度{}超过限度'.format(len(url))
        hashcode = self.geturlhash(url)
        while self._vacob_hash[hashcode] != -1:
            if self._vacob[self._vacob_hash[hashcode]][0] == url:
                return self._vacob_hash[hashcode]
            hashcode = (hashcode + 1) % self._maxUrlNum
        self._vacob_hash[hashcode] = self._count
        self._vacob.append([url, 1])
        self._count += 1
        return self._count - 1



    def searchurl(self, url):
        """查询url的索引"""
        assert len(url) <= MAX_LEN_URL, 'url长度{},超过限度'.format(len(url))
        hashcode = self.geturlhash(url)
        if self._vacob_hash[hashcode] == -1:
            return -1

        count = 0
        while self._vacob[self._vacob_hash[hashcode]][0] != url:
            hashcode = (hashcode + 1) % self._maxUrlNum
            count += 1
            if self._vacob_hash[hashcode] == -1:
                return -1
            if count > self._maxUrlNum:
                print 1
                return -1
        return self._vacob_hash[hashcode]


    @property
    def vacob(self):
        return self._vacob

    def set_vacob(self, vacob):
        self._vacob = vacob


    @property
    def length(self):
        return self._count

    def reduce_vacob(self):
        """去掉频率低的url词汇，保证hashtable稀疏性"""
        print 'the number of url is larger than vocabulary size, exceed reduce_vacob with min_recude %d' \
              % self._min_reduce
        count = 1
        for i in xrange(1, self._count):
            if self._vacob[i][1] > self._min_reduce:
                self._vacob[count] = self._vacob[i]
                count += 1
            else:
                self._vacob[0][1] += self._vacob[i][1]
        #重新计算_vacob_hash
        for i in xrange(self._maxUrlNum):
            self._vacob_hash[i] = -1
        for i in xrange(count):
            hashcode = self.geturlhash(self._vacob[i][0])
            while self._vacob_hash[hashcode] != -1:
                hashcode = (hashcode + 1) % self._maxUrlNum
            self._vacob_hash[hashcode] = i

        self._vacob = self._vacob[0:count]
        self._count = count
        print self._count
        if self._count / (self._maxUrlNum * 0.7) > 0.7:
            self._min_reduce += 1

    def sort_vacob(self):
        # rare_url = self._vacob.pop(0)
        print "sort vacob"
        self._vacob = sorted(self._vacob,
                             key=lambda word: word[1] if word[0] != 'UNK' else INF,
                             reverse=True)
        for i in xrange(self._maxUrlNum):
            self._vacob_hash[i] = -1
        #排除掉低频率的url词汇,并重新生成_vacob_hash
        limit = self._min_reduce if self._min_reduce > self._min_count else self._min_count
        for i in xrange(self._count):
            if self._vacob[i][1] < limit and i > 0:
                x = i
                while x < self._count:
                    self._vacob[0][1] += self._vacob[x][1]
                    x += 1
                self._vacob = self._vacob[0:i]
                self._count = i
                break

            hashcode = self.geturlhash(self._vacob[i][0])
            while self._vacob_hash[hashcode] != -1:
                hashcode = (hashcode + 1) % self._maxUrlNum
            self._vacob_hash[hashcode] = i

    def test_search(self, url):
        # Search Engines Reg
        SousouReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*sousou\.com'
        SouGouReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*sogou\.com'
        Searcg360Reg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*so\.360\.cn/'
        BaiduReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*baidu\.com'
        BingReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*bing\.com'
        AolReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*aol\.com'
        AskReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*ask\.com'
        DaumReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*daum\.net'
        GoogleReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*google\.'
        MailReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*go\.mail\.ru'
        WebCrawlerReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*webcrawler\.com'
        WowReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*us\.wow\.com'
        YahooReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*yahoo\.(com|co)'
        YandexReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*yandex\.(com|by)'
        MySearchReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*zxyt\.cn'
        BingIEReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*bing\.ie'
        SearchLockReg = '^www\.searchlock\.com'
        SoSoReg = '^www\.soso\.com'
        SoReg = '^www\.so\.com'
        GoogleWebLightReg = '^googleweblight\.com'
        result = re.search(SouGouReg + '|' + SousouReg + '|' +Searcg360Reg + '|' + BaiduReg + '|' + BingReg +'|' + AolReg +'|' + AskReg +'|' + DaumReg +'|' +
                             GoogleReg +'|' + MailReg +'|' + WebCrawlerReg +'|' + WowReg +'|' + YahooReg +'|' + YandexReg +'|' + MySearchReg +'|' + BingIEReg +'|' +
                             SearchLockReg +'|' + SoSoReg +'|' + SoReg +'|' + GoogleWebLightReg, url)
        if result:
            return result.group()
        else:
            return None

    def create_url_vacob_from_list(self, urllist, sortable=False):
        """构建url单词表"""
        # #Search Engines Reg
        # SousouReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*sousou\.com'
        # SouGouReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*sogou\.com'
        # Searcg360Reg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*so\.360\.cn/'
        # BaiduReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*baidu\.com'
        # BingReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*bing\.com'
        # AolReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*aol\.com'
        # AskReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*ask\.com'
        # DaumReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*daum\.net'
        # GoogleReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*google\.'
        # MailReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*go\.mail\.ru'
        # WebCrawlerReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*webcrawler\.com'
        # WowReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*us\.wow\.com'
        # YahooReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*yahoo\.(com|co)'
        # YandexReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*yandex\.(com|by)'
        # MySearchReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*zxyt\.cn'
        # BingIEReg = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*bing\.ie'
        # SearchLockReg = r'^www\.searchlock\.com'
        # SoSoReg = r'^www\.soso\.com'
        # SoReg = r'^www\.so\.com'
        # GoogleWebLightReg = r'^googleweblight\.com'

        #Inquiry Pages Reg
        InquirySuccess = r'^www\.made-in-china\.com/sendInquiry/success'
        InquiryFailure = r'^www\.made-in-china\.com/sendInquiry/failure'
        InquiryProd_ = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*made-in-china\.com/(sendInquiry|sendinquiry)/prod_'

        #Multi-search Pages Reg
        multisearch = r'^www\.made-in-china\.com/multi-search/.*/F[0-2]/'
        # multisearchCatalog = r'^www\.made-in-china.com/[^/]*-Catalog/'

        assert urllist, 'url data is empty, check the data'
        if not self._vacob:
            self.addurltovacob('UNK')
            self._epoch_count += 1

        # print 'length{}'.format(len(urllist))

        for i in xrange(len(urllist)):
            if re.search(r'((&|\?)(key)?word=)|(^membercenter\.)', urllist[i]) is not None:
                # print urllist[i]
                self._vacob[0][1] += 1
                continue
            # Unify all external Search engines to External.Search.Engines
            # if re.search(SouGouReg | SousouReg |Searcg360Reg | BaiduReg | BingReg | AolReg | AskReg | DaumReg |
            #         GoogleReg | MailReg | WebCrawlerReg | WowReg | YahooReg | YandexReg | MySearchReg | BingIEReg |
            #         SearchLockReg | SoSoReg | SoReg | GoogleWebLightReg, urllist[i]) is not None:
            t = self.test_search(urllist[i])
            if t is not None:
                urllist[i] = t

            # Unify all sendInquiry/success pages  to sendInquiry/success
            if re.search(InquirySuccess, urllist[i]) is not None:
                urllist[i] = r'www.made-in-china.com/sendInquiry/success'

            # Unify all sendInquiry/failure pages to sendInquiry/failure
            if re.search(InquiryFailure, urllist[i]) is not None:
                urllist[i] = r'www.made-in-china.com/sendInquiry/failure'

            # Unify all senInquiry/prods_ pages tp sendInqiry/prod_
            if re.search(InquiryProd_, urllist[i]) is not None:
                urllist[i] = r'www.made-in-china\.com/sendInquiry/prods_item.html'

            # Unify all multi-search/.*F1 pages to www\.made-in-china\.multi-search/item/F1/pages\.html
            if re.search(multisearch, urllist[i]) is not None:
                urllist[i] = r'www.made-in-china.com/multi-search/items/F1/pages.html'
            # if re.search(multisearchCatalog, urllist[i]) is not None:
            #     urllist[i] = r'www.made-in-china.com/multi-search/items-Catalog/F0.html'

            x = self.searchurl(urllist[i])
            if x == -1:
                self.addurltovacob(urllist[i])
            else:
                self._vacob[x][1] += 1
            if self._count > (self._maxUrlNum * 0.7):
                print self._count
                if not sortable:
                    self.sort_vacob()
                self.reduce_vacob()
            self._epoch_count += 1
        #对url词典进行降序排序
        if sortable:
            print 'start sort the vocabulary'
            self.sort_vacob()
            print 'building url vocabulary end'
        # print self._vacob[0:100]
        return self._vacob

    @property
    def epoch_count(self):
        return self._epoch_count


    def set_epoch_count(self, n):
        self._epoch_count = n

    def save_vacob(self):
        pass

    def read_vacob(self):
        pass
    # def get_vacob(self):


