import torch
import os
import pandas as pd


class knn_torch:
    def __init__(self, datafile=None, save_file='knn_data.pth'):
        self.x_data = None
        self.y_data = None
        self.save_file = save_file
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

        if self.x_data is None:
            print('No trained classes found')
            return None

        if len(x.shape) == 1:

            dist = torch.norm(self.x_data - x, dim=1, p=None)

            knn = dist.topk(5, largest=False)
            # print(knn, knn.indices)
            # nearest_idx = knn.indices[0]

            # cl = self.y_data[nearest_idx]
            cl = [self.y_data[i] for i in knn.indices]
            # print(cl)
            return cl
        elif len(x.shape) == 2:
            clss = []
            for x_el in x:

                dist = torch.norm(self.x_data - x_el, dim=1, p=None)
                knn = dist.topk(1, largest=False)
                nearest_idx = knn.indices[0]

                # for d, gt in zip(dist, self.y_data):
                #     print(d.data, gt)

                # print(knn[0].data, nearest_idx)

                cl = self.y_data[nearest_idx]
                clss.append(cl)
            return clss
