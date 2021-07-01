import torch
import os
import pandas as pd
import time
# from statistics import mode
from scipy import stats as s

class knn_torch:
    def __init__(self, datafile=None, savefile=None):
        self.x_data = None
        self.y_data = None
        self.save_file = datafile if not savefile else savefile
        self.classes from scipy import stats as s('File found')
                data = torch.load(datafile)
                self.x_data = data['x']
                self.y_data = data['y']
                print(
                    f'Found {self.x_data.shape[0]} points with {len(set(self.y_data))} classes')
                print(pd.Series(self.y_data).value_counts())
                self.classes = list(set(self.y_data))

                if torch.cuda.is_available():
                    self.x_data = self.x_data.cuda()
            else:
                print('File not found')

    def add_points(self, x, y):

        print(x.shape, len(y))
        if self.x_data == None:
            self.x_data = x
            self.y_data = y
        else:
            self.x_data = torch.cat([self.x_data, x])
            self.y_data = self.y_data + y
        self.classes = list(set(self.y_data))
        # print(x.shape, self.x_data.shape)
        torch.save({'x': self.x_data.detach().cpu(),
                    'y': self.y_data}, self.save_file)

    def classify(self, x):
        print(x.shape)

        if self.x_data is None:
            print('No trained classes found')
            return None
        # print(self.x_data.shape, x.shape)

        # if len(x.shape) == 1:

        #     dist = torch.norm(self.x_data - x, dim=1, p=None)

        #     knn = dist.topk(10, largest=False)
        #     # print(knn, knn.indices)
        #     # nearest_idx = knn.indices[0]

        #     # cl = self.y_data[nearest_idx]
        #     cl = [self.y_data[i] for i in knn.indices]
        #     return cl

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # if len(x.shape) == 2:
        clss = []
        confs = []
        for x_el in x:
            knn_size = 20
            dist = torch.norm(self.x_data - x_el, dim=1, p=None)
            knn = dist.topk(knn_size, largest=False)
            # nearest_idx = knn.indices[0]

            near_y = list(map(self.y_data.__getitem__, knn.indices))

            # cl = mode(near_y)
            cl = s.mode(near_y)[0]

            frac = near_y.count(cl) / knn_size

            # print(near_y)
            # print(mode(near_y))
            # print(near_y.count(mode(near_y)) / knn_size)
            # exit()

            # for d, gt in zip(dist, self.y_data):
            #     print(d.data, gt)

            # print(knn[0].data, nearest_idx)

            # cl = self.y_data[nearest_idx]
            # cl = mode(near_y)
            clss.append(cl)
            confs.append(frac)
        return clss, confs
