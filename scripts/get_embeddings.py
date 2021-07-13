import cv2 as cv
import os
import torch
from matplotlib import pyplot as plt


from models.feature_extractor import image_embedder
from models.knn import *

import re

torch.set_grad_enabled(False)

def get_embeddings(folder, model):
    embs = None
    labels = []

    for (root, dirs, files) in os.walk(folder):
        # print([root, files])
        for f in files:
            if f.endswith('.png'):
                im = cv.imread(root + '/' + f, cv.IMREAD_COLOR)
                im = cv.cvtColor(im, cv.COLOR_BGR2RGB)


                # plt.imshow(im)
                # plt.show()

                emb = model([im])

                if embs == None:
                    embs = emb.squeeze(
                    ).unsqueeze(0)
                else:
                    embs = torch.cat(
                        [embs, emb.squeeze().unsqueeze(0)])
                label = f.split('_')[0]
                labels.append(label)

                print(embs.shape)
    return embs, labels



if __name__ == '__main__':
    image_folder = 'saved_masks'

    emb_size = 64
    postfix = '_100cl'

    folder = f'/home/iiwa/Nenakhov/metric_learning/example_saved_models/mobilenetv3_small_{emb_size}{postfix}'
    fs = os.listdir(folder)

    r = re.compile("embedder_best")
    emb_file = folder + '/' + list(filter(r.match, fs))[0]

    r = re.compile("trunk_best")
    trunk_file = folder + '/' + list(filter(r.match, fs))[0]

    model = image_embedder(trunk_file=trunk_file, emb_file=emb_file, emb_size=emb_size)

    clf = knn_torch(
            datafile=f'datafiles/test_data_own_{emb_size}{postfix}.pth')


    embs = None
    labels = []

    for (root, dirs, files) in os.walk(image_folder):
        # print([root, files])
        for f in files:
            if f.endswith('.png'):
                im = cv.imread(root + '/' + f, cv.IMREAD_COLOR)
                im = cv.cvtColor(im, cv.COLOR_BGR2RGB)


                # plt.imshow(im)
                # plt.show()

                emb = model([im])

                if embs == None:
                    embs = emb.squeeze(
                    ).unsqueeze(0)
                else:
                    embs = torch.cat(
                        [embs, emb.squeeze().unsqueeze(0)])
                label = f.split('_')[0]
                labels.append(label)

                print(embs.shape)



                # print(emb.shape)

    clf.add_points(embs, labels)




        

