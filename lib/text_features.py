import collections
import math
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from lib.loader import Loader
import numpy as np


def merge(words, sep=" "):
    if type(words) is str:
        return words
    final = ""
    for w in words:
        final += w + sep
    return final[:-len(sep)]


def get_gist_text(text: str):
    if type(text) is not str:
        return ""
    regex = re.compile('[^a-zA-Z \n]')
    step1 = regex.sub('', text)
    step2 = re.split(' ', step1)
    step3 = list(filter(lambda x: len(x) < 25, step2))
    step4 = set(step3)
    clean_text = list(step4)
    return merge(clean_text)


# TODO : Set ops
def top_words(text_ls: list):
    total = list()
    for t in text_ls:
        clean_text = get_gist_text(t)
        total += clean_text

    return collections.Counter(total)





def make_buckets(text_ls: list, lb_ls: list, num_buks):
    buckets = dict()
    for idx in range(num_buks):
        buckets[idx] = list()
    for idx in range(len(text_ls)):
        buckets[lb_ls[idx]].append(text_ls[idx])
    return buckets


def test_buckets():
    buks = make_buckets(texts, labels, 6)
    sets = list()
    for key in buks:
        top = top_words(buks[key])
        print(key, top.most_common(5))
        # all_words = list(map(lambda x: x[0], top.most_common(50)))
        sets.append(set(top))
    intersect = sets[0]
    uni = list()
    for s in sets:
        intersect = s.intersection(intersect)
    for s in sets:
        uni.append(s.difference(intersect))


if __name__ == '__main__':
    train_data = Loader("../kaggle/train.csv", [], ["id", "description", "label"])
    IDs = train_data.raw["id"].values.tolist()
    texts = train_data.raw["description"].values.tolist()
    labels = train_data.raw["label"].values.tolist()

    # IDs = IDs[:550]
    # texts = texts[:550]
    # labels = labels[:550]

    gists = list(map(lambda x: merge(get_gist_text(x)), texts))
    X_train, X_test, y_train, y_test = train_test_split(gists, labels, test_size=0.2)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(X_train)

    target = 500
    reduced_x = list()
    for idx, i in enumerate(X):
        info = "Progress {curr}/{total}".format(curr=idx, total=len(X_train))
        print("\r" + info, end="")
        i = i.toarray()[0]
        anchor = 0
        limit = math.floor(len(i)/target)
        summ = 0
        red = list()
        for j in i:
            anchor += 1
            if anchor % limit == 0:
                red.append(summ/limit)
                summ = 0
            else:
                summ += j
        if summ != 0:
            red.append(summ / limit)
        if len(red) < 500:
            for mk in range(500 - len(red)):
                red.append(np.random.rand())
        elif len(red) > 500:
            for mk in range(len(red) - 500):
                red.pop(0)
        reduced_x.append(red)
    # k_model = KNeighborsClassifier(n_neighbors=6, weights='distance')
    k_model = GaussianMixture(n_components=6)
    k_model.fit(X)

    # k_model.fit(X, y_train)
    x_test = vectorizer.transform(X_test)
    x_test = x_test.toarray()
    pred = k_model.predict(x_test)
    error = 0
    dist = 0
    for i, p in enumerate(pred):
        if p != y_test[i]:
            error += 1
            dist += abs(p - y_test[i])
    print(1 - error/len(pred))
    print(dist/len(pred))
