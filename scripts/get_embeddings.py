import cv2 as cv
import os
import torch
from matplotlib import pyplot as plt


from models.feature_extractor import image_embedder
from models.knn import *


torch.set_grad_enabled(False)

if __name__ == '__main__':
    folder = '/home/iiwa/Nenakhov/coil-100'

    model = image_embedder('mobilenetv3_small_128_models.pth')

    clf = knn_torch(
            datafile='test_data_coil-100.pth')


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



                # print(emb.shape)

    clf.add_points(embs, labels)


        

