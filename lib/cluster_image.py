#!/usr/bin/python3

import glob
import math
import os
import os.path
import shutil
import statistics

import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19 as NNModel
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from sklearn.cluster import MiniBatchKMeans as KMeans

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


class ImageCluster:

    def __init__(self, src, k, cluster_home, train_IDs, test_IDs, labels):
        self.cluster_home = cluster_home + str(k) + "/"
        self.org_images = src
        self.train_ids = train_IDs
        self.test_IDs = test_IDs

        self.k = k
        self.mapping = dict()
        self.clusters = dict()

        if not os.path.exists(self.cluster_home):
            self.cluster_images()
        self.fetch_cluster()
        self.cavg = self.cluster_avg(self.train_ids, labels)

    def fetch_cluster(self):
        image_paths = os.listdir(self.cluster_home)
        for ip in image_paths:
            img_base = os.path.basename(ip)
            cluster_num = int(img_base[0])
            img_id = int(((img_base.split("_"))[1]).split(".")[0])
            self.mapping[img_id] = cluster_num
            self.clusters[cluster_num] = img_id

    def cluster_images(self):
        image.LOAD_TRUNCATED_IMAGES = True
        model = NNModel(weights='imagenet', include_top=False)

        filelist = glob.glob(os.path.join(self.org_images, '*.jpg'))
        filelist.sort()
        featurelist = []
        for i, imagepath in enumerate(filelist):
            info = "Progress {curr}/{total}".format(curr=i, total=len(filelist))
            print("\r" + info, end="")

            # noinspection PyBroadException
            try:
                img = image.load_img(imagepath, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                features = np.array(model.predict(img_data, batch_size=1000))
                featurelist.append(features.flatten())
            except:
                continue

        # Clustering
        to_fit = np.array(featurelist)
        kmeans = KMeans(n_clusters=self.k, random_state=0, verbose=1).fit(to_fit)

        # This is a computation heavy task so I'm copying the images renamed according to cluster for the next time.
        try:
            os.makedirs(self.cluster_home)
        except OSError:
            pass

        for i, m in enumerate(kmeans.labels_):
            shutil.copy(filelist[i], self.cluster_home + str(m) + "_" + os.path.basename(filelist[i]))

        return kmeans

    def cluster_avg(self, ids, labels):
        comparison_dict = dict()
        for idx in range(len(ids)):
            comparison_dict[ids[idx]] = labels[idx]

        image_ids = self.mapping.keys()
        image_ids = list(map(lambda x: int(x), image_ids))
        image_ids = list(set(image_ids).intersection(set(ids)))

        score_meat = dict()
        score_meat["id"] = []
        score_meat["cluster"] = []
        score_meat["label"] = []
        for idx in range(len(image_ids)):
            iid = image_ids[idx]
            c = self.mapping[iid]
            label = comparison_dict[iid]
            score_meat["id"].append(iid)
            score_meat["cluster"].append(c)
            score_meat["label"].append(label)

        avg = self.cluster_evaluation(score_meat["label"], score_meat["cluster"])
        return avg

    def image_score(self, img_ID):
        if img_ID not in self.mapping:
            return 0
        pred = self.mapping[img_ID]
        soft_max_sum = 0
        for pa in self.cavg:
            soft_max_sum += math.exp(self.cavg[pa])
        return int((math.exp(pred) / soft_max_sum) * 100)

    def cluster_evaluation(self, real, pred):
        cl_avg = dict()
        for i in range(self.k):
            cl_avg[i] = list()
        for i in range(len(real)):
            cl_avg[pred[i]].append(real[i])
        for i in range(self.k):
            if len(cl_avg[i]) == 0:
                cl_avg[i] = 0
            else:
                cl_avg[i] = statistics.mean(cl_avg[i])
        return cl_avg
