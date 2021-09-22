from torch import nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from mlp import MLP
import torchvision
from pytorch_metric_learning.utils import common_functions


class image_embedder(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # filedata = torch.load(file)
        self.trunk = torchvision.models.mobilenet_v3_small()
        self.trunk.classifier = common_functions.Identity()
        self.embedder = MLP([576, 128])

        self.trunk.load_state_dict(
            torch.load('/home/ivan/grasping/metric_learning/example_saved_models/mobilenetv3_small_128/trunk_5.pth'))

        self.embedder.load_state_dict(torch.load(
            '/home/ivan/grasping/metric_learning/example_saved_models/mobilenetv3_small_128/embedder_5.pth'))

        self.trunk.to(self.device)
        self.embedder.to(self.device)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def forward(self, x):

        if isinstance(x, list):
            x = torch.stack([self.transforms(i) for i in x])

        x = x.to(self.device)

        res = self.trunk(x)

        if self.embedder:
            res = self.embedder(res)
            return res

    # def load_from_file(self, filename):
    #     self.backbone, self.embedder = torch.load(filename)


if __name__ == '__main__':

    trunk = torchvision.models.mobilenet_v3_small()
    trunk.classifier = common_functions.Identity()

    # m = image_embedder()
    ar = torch.ones([5, 3, 224, 224])

    # print(m(ar))
    tr_res = trunk(ar)
    print(tr_res.shape)

    mlp = MLP([576, 128])

    ar = torch.ones([3, 576])
    # print(ar.shape)

    print(mlp(tr_res).shape)
