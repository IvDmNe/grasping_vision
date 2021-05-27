import torch


class knn_torch:
    def __init__(self, datafile=None, save_file='knn_data.pth'):
        self.x_data = None
        self.y_data = None
        self.save_file = save_file

        if datafile:
            data = torch.load(datafile)
            self.x_data = data['x']
            self.y_data = data['y']
            if torch.cuda.is_available():
                self.x_data = self.x_data.cuda()

    def add_points(self, x, y):
  
        if self.x_data == None:
            self.x_data = x
            self.y_data = y
        else:
            self.x_data = torch.cat([self.x_data, x])
            self.y_data = self.y_data + y

        # print(type(self.x_data))
        # print(type(self.y_data))
        torch.save({'x': self.x_data.detach().cpu(),
                    'y': self.y_data}, self.save_file)

    def classify(self, x):
        
        if len(x.shape) == 1:
            dist = torch.norm(self.x_data - x, dim=1, p=None)

            knn = dist.topk(1, largest=False)

            nearest_idx = knn.indices[0]

            cl = self.y_data[nearest_idx]
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
