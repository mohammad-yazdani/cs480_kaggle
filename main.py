#!/usr/bin/python3
import datetime
import os

import numpy as np
import pandas
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.utils import np_utils

from lib.cluster_image import ImageCluster
from lib.cluster_text import TextCluster
from lib.loader import Loader
from lib.score import Score


def baseline_model():
    num_out = 6
    model = keras.Sequential()

    model.add(layers.Conv1D(128, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.Conv1D(128, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.Conv1D(128, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.Conv1D(128, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(96, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.Conv1D(96, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.Conv1D(96, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(64, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.Conv1D(64, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(32, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.Conv1D(32, 1, input_shape=(26, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(num_out, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    train_file = "kaggle/train.csv"
    if not os.path.exists(train_file):
        print("Train file " + train_file + " not found.")
        exit(1)

    test_file = "kaggle/test.csv"
    if not os.path.exists(test_file):
        print("Test file " + test_file + " not found.")
        exit(1)

    image_src = "kaggle/thumbnails/thumbnails"
    if not os.path.exists(image_src):
        print("Image directory " + image_src + " not found.")
        exit(1)

    cluster_home_prefix = "data/clusters"
    thumbnail_K = 10
    description_K = 6

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
        "description",
        # "links",
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

    # Data prep
    scorer = Score("data/tag_scores.csv")
    train_IDs = Loader(csv_file=train_file, split_pairs=[], filter_headers=["id"])
    train_IDs = train_IDs.raw["id"].values.tolist()
    test_IDs = Loader(csv_file=test_file, split_pairs=[], filter_headers=["id"])
    test_IDs = test_IDs.raw["id"].values.tolist()

    train_data = Loader(csv_file=train_file, split_pairs=[], filter_headers=filters)
    train_data.apply_category_encoding()
    train_data.map_column("link-tags", scorer.score_tag)

    # Description clustering
    train_labels = train_data.raw["label"].values.tolist()
    train_descriptions = train_data.raw["description"].values.tolist()
    tc = TextCluster(train_descriptions, train_labels, description_K)
    train_data.map_column_to_new("description", "text_score", tc.score_text, 0)
    train_data.map_column_to_new("description", "url_score", TextCluster.find_url, 0)
    train_data.remove_column("description")

    # Thumbnail clustering
    cluster_master = ImageCluster(image_src, thumbnail_K, cluster_home_prefix, train_IDs, test_IDs, train_labels)
    train_data.map_column_to_new("id", "visual_score", cluster_master.image_score, 0)
    train_data.remove_column("id")

    train_raw = train_data.raw
    X = train_raw.iloc[:, :-1].values
    y = train_raw.iloc[:, 26].values

    X = np.expand_dims(X, axis=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    X_train = np.asarray(X_train, dtype=np.float)
    y_train = np.asarray(y_train, dtype=np.int)
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    test_data = Loader(csv_file=test_file, split_pairs=[], filter_headers=filters[:-1])
    test_data.apply_category_encoding()
    test_data.map_column("link-tags", scorer.score_tag)
    test_data.map_column_to_new("description", "text_score", tc.score_text, 0)
    test_data.map_column_to_new("description", "url_score", TextCluster.find_url, 0)
    test_data.remove_column("description")
    test_data.map_column_to_new("id", "visual_score", cluster_master.image_score, 0)
    test_data.remove_column("id")

    test_raw = test_data.raw
    X_test_real = test_raw
    X_test_real = np.asarray(X_test_real, dtype=np.float)
    X_test_real = np.expand_dims(X_test_real, axis=2)

    # Predict 50, 50 gave 93
    estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=50, verbose=1)
    kfold = KFold(n_splits=10, shuffle=True)
    print(len(X_train))
    results = cross_val_score(estimator, X_train, dummy_y, cv=kfold)
    # results = cross_val_score(estimator, X_train, dummy_y)

    # Local test
    estimator.fit(X_train, y_train)
    local_pred = estimator.predict(X_test)
    precision, recall, fscore, support = score(y_test, local_pred)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    errors = 0
    for idx in range(len(y_test)):
        if local_pred[idx] != y_test[idx]:
            errors += 1
    print(1 - (errors / len(y_test)))
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    # Submission
    estimator.fit(X_train, y_train)
    real_pred = estimator.predict(X_test_real)

    test_ids = Loader(csv_file=test_file, split_pairs=[], filter_headers=["id"])
    test_ids = test_ids.raw.values.tolist()
    IDs = list()
    for idx in test_ids:
        IDs.append(idx[0])

    result = pandas.DataFrame(data={"id": IDs, "label": real_pred})
    ts = str(int(datetime.datetime.now().timestamp()))
    result.to_csv("./submission" + ts + ".csv", sep=',', index=False)
