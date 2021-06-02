from torch import nn
import torch.nn.functional as F


class feature_extractor(nn.Module):

    def __init__(self, backbone):
        super(feature_extractor, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        feats = self.backbone(x)
        output = F.max_pool2d(feats, kernel_size=feats.size()[2:])
        output = output.squeeze()
        return output
