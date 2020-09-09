#!/usr/bin/python3

import math
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cluster_home = "data/text_clusters"


class TextCluster:

    def __init__(self, texts, labels, k):
        self.X = TextCluster.clean(texts)
        self.y = labels
        self.k = k

        self.buckets = self.make_buckets()
        self.score_meta = self.get_score_meta()

    def make_buckets(self):
        X = self.X
        y = self.y

        combo = list()
        for idx in range(len(X)):
            combo.append([str(X[idx]), int(y[idx])])

        buckets = dict()
        buckets["0"] = list(filter(lambda x: len(x[0]) <= 0, combo))
        buckets["50"] = list(filter(lambda x: 50 >= len(x[0]) > 0, combo))
        buckets["100"] = list(filter(lambda x: 100 >= len(x[0]) > 50, combo))
        buckets["250"] = list(filter(lambda x: 250 >= len(x[0]) > 100, combo))
        buckets["500"] = list(filter(lambda x: 500 >= len(x[0]) > 250, combo))
        buckets["large"] = list(filter(lambda x: len(x[0]) > 500, combo))

        return buckets

    def get_score_meta(self):
        accuracy = dict()
        for key in self.buckets:
            if str(key) == "0":
                accuracy[key] = 0
                continue
            labels = list(map(lambda x: x[1], self.buckets[key]))
            if len(labels) == 0:
                continue
            descs = list(map(lambda x: x[0], self.buckets[key]))
            accu = self.test_cluster(descs, labels, self.k)

            accuracy[key] = math.exp(accu * 10)
        return accuracy

    def score_text(self, text):
        if type(text) is not str:
            return 0
        text_len = len(text)
        if text_len > 500:
            return self.score_meta["large"]
        for key in self.score_meta:
            if key == "large":
                continue
            if text_len <= int(key):
                return self.score_meta[key]
        return None

    @staticmethod
    def test_cluster(buk_X, buk_y, k):
        X = buk_X
        y = buk_y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        pred = TextCluster.run_model(X_train, y_train, k, X_test)
        errors = 0
        pred = list(pred)
        y_test = list(y_test)
        for idx in range(len(pred)):
            errors += (pred[idx] != y_test[idx])

        acc = 1 - (errors / len(y_test))
        return acc

    @staticmethod
    def run_model(Descriptions, Labels, true_k, test):
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(Descriptions)
        k_model = KNeighborsClassifier(n_neighbors=true_k, weights='distance')
        k_model.fit(X, Labels)
        x_test = vectorizer.transform(test)
        pred = k_model.predict(x_test)
        return pred

    @staticmethod
    def find_url(string):
        if type(string) is not str:
            return 0
        regex = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|io)\b/?(?!@)))"
        url = re.findall(regex, string)
        return len(url)

    @staticmethod
    def clean(desc):
        clean_ds = list()

        for idx in range(len(desc)):
            d = desc[idx]
            if type(d) is float or d == "" or type(d) is not str:
                d = ""
            clean_ds.append(d)

        return clean_ds
