import torch
import os
import pandas as pd
import time
from statistics import mode
from scipy import stats as s
from scipy.spatial.distance import cdist
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn import svm


class knn_torch:
    def __init__(self, datafile=None, savefile=None, knn_size=10):

        self.knn_size = knn_size
        self.x_data = None
        self.y_data = None
        self.save_file = datafile if not savefile else savefile
        self.classes = None

        # self.outlier_estimator = LocalOutlierFactor(metric='minkowski', novelty=True)
        # self.outlier_estimator = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

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

                # self.outlier_estimator.fit(self.x_data.cpu())
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

        # self.outlier_estimator.fit(self.x_data.cpu())


    def remove_class(self, cl):
        inds_to_keep = [idx for idx, el in enumerate(self.y_data) if el != cl]

        self.x_data = self.x_data[inds_to_keep]
        self.y_data = [self.y_data[i] for i in inds_to_keep]

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


            x_el = x_el.unsqueeze(0)
            dist = cdist(x_el.cpu(), self.x_data.cpu(), metric='cosine').squeeze()
            dist = torch.Tensor(dist)

            # dist = torch.norm(self.x_data - x_el, dim=1, p=None)

            # print(dist.min(), dist.max())


            knn = dist.topk(self.knn_size, largest=False)


            near_y = list(map(self.y_data.__getitem__, knn.indices))
            cl = s.mode(near_y)[0]


            frac = near_y.count(cl) / self.knn_size

            clss.append(cl[0])
            confs.append(frac)

            # print(self.outlier_estimator.n_samples_fit_, self.outlier_estimator.n_features_in_)

            # smallest_dist = self.outlier_estimator.predict(x_el.cpu())[0]
            # print(smallest_dist)

            # smallest_dist = dist[knn.indices[0]]
            # print(dist.min(), dist.max())
            # avg_dist = dist[knn.indices].max()
            min_dists.append(dist.min())


        return clss, confs, min_dists
