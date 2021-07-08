from torch import nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from models.mlp import MLP
import torchvision
# from pytorch_metric_learning.utils import common_functions


class feature_extractor(nn.Module):

    def __init__(self, backbone):
        super(feature_extractor, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        feats = self.backbone(x)
        output = F.max_pool2d(feats, kernel_size=feats.size()[2:])
        output = output.squeeze()
        return output


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class image_embedder(nn.Module):
    def __init__(self, file):
        super().__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.trunk = torchvision.models.mobilenet_v3_small()
        self.trunk.classifier = Identity()
        self.embedder = MLP([576, 128])

        self.trunk.load_state_dict(
            torch.load('models/trunk_1.pth'))

        self.embedder.load_state_dict(torch.load(
            'models/embedder_1.pth'))

        self.trunk.to(self.device)
        self.embedder.to(self.device)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.eval()

    def forward(self, x):

        if isinstance(x, list):
            x = torch.stack([self.transforms(i) for i in x])
        else:
            x = self.transforms(x)

        x = x.to(self.device)

        res = self.trunk(x)

        if self.embedder:
            res = self.embedder(res)
        return res

    # def load_from_file(self, filename):
    #     self.backbone, self.embedder = torch.load(filename)
