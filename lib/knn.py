#!/usr/bin/python3
import datetime
import os

from lib.loader import Loader

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    train_file = "kaggle/train.csv"
    if not os.path.exists(train_file):
        print("Train file " + train_file + " not found.")
        exit(1)

    test_file = "kaggle/test.csv"
    if not os.path.exists(test_file):
        print("Test file " + test_file + " not found.")
        exit(1)

    category_splitters = {"category": ["jam", "compo"]}
    splitter = category_splitters
    filters = [
        "id",
        "competition-num",
        "category",
        "num-comments",
        "feedback-karma",
        "ratings-given",
        "ratings-received",
        "links",
        "link-tags",
        "num-authors",
        "prev-games",
        "fun-average",
        "innovation-average",
        "theme-average",
        "graphics-average",
        "audio-average",
        "humor-average",
        "mood-average",
        "fun-rank",
        "innovation-rank",
        "theme-rank",
        "graphics-rank",
        "audio-rank",
        "humor-rank",
        "mood-rank",
        "label"
    ]

    # train_data = Loader(csv_file=train_file, split_pairs=splitters, filter_headers=filters)
    # print(train_data.data)

    knn_filters = [
        "category",
        "num-comments",
        "feedback-karma",
        "ratings-given",
        "ratings-received",
        "num-authors",
        "prev-games",
        "fun-average",
        "innovation-average",
        "theme-average",
        "graphics-average",
        "audio-average",
        "humor-average",
        "mood-average",
        "fun-rank",
        "innovation-rank",
        "theme-rank",
        "graphics-rank",
        "audio-rank",
        "humor-rank",
        "mood-rank",
        "label"
    ]

    train_data = Loader(csv_file=train_file, split_pairs=splitter, filter_headers=knn_filters)
    df = train_data.splits["category"]["jam"]  # TODO : Split here

    X = df.iloc[:, :-1].values
    y = df.iloc[:, 20].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    test_data = Loader(csv_file=test_file, split_pairs=splitter, filter_headers=knn_filters[:-1])
    real_X_test = test_data.splits["category"]["jam"]  # TODO : Split here

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    real_X_test = scaler.transform(real_X_test)

    error = []
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    min_error = min(error)
    best_nn = 0
    for idx in range(len(error)):
        if error[idx] == min_error:
            best_nn = idx

    classifier = KNeighborsClassifier(n_neighbors=best_nn)
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    plt.hist(pred, bins=100)
    plt.show()

    # TODO : REAL shit
    real_pred = classifier.predict(real_X_test)
    # plt.hist(pred, bins=100)
    # plt.show()

    test_ids = Loader(csv_file=test_file, split_pairs=[], filter_headers=["id"])
    test_ids = test_ids.raw.values.tolist()
    IDs = list()
    for idx in test_ids:
        IDs.append(idx[0])

    print(len(real_pred), len(IDs))
    result = pandas.DataFrame(data={"id": IDs, "label": real_pred})
    ts = str(int(datetime.datetime.now().timestamp()))
    result.to_csv("./submission" + ts + ".csv", sep=',', index=False)
