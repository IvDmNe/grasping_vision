import torch
import os
import pandas as pd
import time
from statistics import mode
from scipy import stats as s


class knn_torch:
    def __init__(self, datafile=None, savefile=None):
        self.x_data = None
        self.y_data = None
        self.save_file = datafile if not savefile else savefile
        self.classes = None

        if datafile:
            print(f'loading data from file: {datafile}')
            if (os.path.exists(datafile)):
                print('File found')
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

        # print(x.shape, len(y))
        if self.x_data == None:
            self.x_data = x
            self.y_data = y
        else:
            self.x_data = torch.cat([self.x_data, x])
            self.y_data = self.y_data + y
        self.classes = list(set(self.y_data))

        torch.save({'x': self.x_data.detach().cpu(),
                    'y': self.y_data}, self.save_file)

    def classify(self, x):

        if self.x_data is None:
            print('No trained classes found')
            return None


        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        clss = []
        confs = []
        min_dists = []
        for x_el in x:
            knn_size = 20
            dist = torch.norm(self.x_data - x_el, dim=1, p=None)
            knn = dist.topk(knn_size, largest=False)

            # print(dist[knn.indices[0]])

            smallest_dist = dist[knn.indices[0]]
            min_dists.append(smallest_dist)

            near_y = list(map(self.y_data.__getitem__, knn.indices))
            cl = s.mode(near_y)[0]
            frac = near_y.count(cl) / knn_size

            clss.append(cl[0])
            confs.append(frac)
        return clss, confs, min_dists
