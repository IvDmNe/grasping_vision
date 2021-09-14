#!/home/ivan/anaconda3/bin/python

# this project packages
from models.segmentation_net import *
from models.feature_extractor import image_embedder, dino_wrapper
from models.mlp import MLP
from models.knn import *
from utilities.utils import *

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge

# PyTorch
import torch
from torch.nn import functional as F

# Detectron 2
from detectron2.utils.logger import setup_logger
from detectron2.utils.colormap import colormap

# misc
import threading
import cv2 as cv
import time
import numpy as np

# Pytorch metric learning
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import re
from tqdm import tqdm
from umap import UMAP
import seaborn as sns


torch.set_grad_enabled(False)


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(data, accuracy_calculator):

    # train_embeddings, train_labels = get_all_embeddings(train_set, model)
    # data = torch.load(datafile)
    embs = data['x']
    labels = data['y']

    enc_labels = LabelEncoder().fit_transform(labels)

    train_embs, test_embs, train_labels, test_labels = train_test_split(embs, enc_labels, test_size=0.4, random_state=42)
    # test_embeddings, test_labels = get_all_embeddings(test_set, model)
    # print("Computing accuracy")
    
    accuracies = accuracy_calculator.get_accuracy(test_embs, 
                                                train_embs,
                                                test_labels,
                                                train_labels,
                                                False)
    # print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

    # save test results
    return accuracies["precision_at_1"]

  
def get_embeddings(folder, model):
    embs = None
    labels = []

    with tqdm() as t:

        for (root, dirs, files) in os.walk(folder):
            for f in files:
                if f.endswith('.png'):
                    im = cv.imread(root + '/' + f, cv.IMREAD_COLOR)
                    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

                    emb = model([im])

                    if embs == None:
                        embs = emb.squeeze(
                        ).unsqueeze(0)
                    else:
                        embs = torch.cat(
                            [embs, emb.squeeze().unsqueeze(0)])
                    label = f.split('_')[0]
                    labels.append(label)
                    t.update()

        
    return {'x': embs, 'y': labels}


def visualize_embeddings(data, emb_sz, postfix=None, arch=None):

    time_start = time.time()
    umap = UMAP()


    print(data['x'].shape, len(data['y']))

    results = umap.fit_transform(data['x'].squeeze().cpu())

    print('visualization done! Time elapsed: {} seconds'.format(time.time()-time_start))


    labels = data['y']

    fig = plt.figure(figsize=(16, 10))

    ax = sns.scatterplot(
        x=results[:, 0], y=results[:, 1],
        hue=data['y'],
        palette=sns.color_palette("hls", len(set(data['y']))),
        # data=df_subset,
        legend="full",
        # alpha=0.3
    )
    ax.set_title(f'visuals/vis_results_{emb_sz}{postfix}_{arch}')
    # plt.savefig(f'visuals/vis_results_{emb_sz}{postfix}_{arch}.png')

    plt.show()

if __name__ == '__main__':

    postfix = '_100cl'
    arch = 'mobilenetv3_small'

    

    for emb_size in [128]:#, 64, 128]:

        folder = f'/home/iiwa/Nenakhov/metric_learning/example_saved_models/{arch}_{emb_size}{postfix}'
        fs = os.listdir(folder)

        r = re.compile("embedder_best")
        emb_file = folder + '/' + list(filter(r.match, fs))[0]

        r = re.compile("trunk_best")
        trunk_file = folder + '/' + list(filter(r.match, fs))[0]

        # print(trunk_file)


        model = image_embedder(trunk_file=trunk_file, emb_file=emb_file, emb_size=emb_size)

    


        data = get_embeddings('saved_masks', model)

        
        acc = test(data, AccuracyCalculator(include = ("precision_at_1",), k=10))

        with open('test_results', 'a+') as f:
            f.write(f'emb_sz: {emb_size}, acc: {acc:.3f}{postfix}_{arch}\n')
            print(f'emb_sz: {emb_size}, acc: {acc:.3f}{postfix}_{arch}\n')


        visualize_embeddings(data, emb_size, postfix=f'{postfix}', arch=arch)


    model = dino_wrapper()

    data = get_embeddings('saved_masks', model)

    emb_size = 384
    postfix = ''
    arch = 'DINO'

        
    acc = test(data, AccuracyCalculator(include = ("precision_at_1",), k=5))

    with open('test_results', 'a+') as f:
        f.write(f'emb_sz: {emb_size}, acc: {acc:.3f}{postfix}_{arch}\n')
        print(f'emb_sz: {emb_size}, acc: {acc:.3f}{postfix}_{arch}\n')


    visualize_embeddings(data, emb_size, postfix='', arch='DINO')

