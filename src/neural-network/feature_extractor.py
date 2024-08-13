import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvggish import vggish, vggish_input


class VGGishFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGishFeatureExtractor, self).__init__()
        self.vggish = vggish(pretrained)
        # Remove the last layer (classification layer)
        self.vggish.postprocess = nn.Identity()
        self.vggish.embeddings = nn.Identity()

    def forward(self, x):
        # x is expected to be a batch of spectrograms: (batch_size, 1, time, frequency)
        # VGGish expects input in the shape (batch_size, 1, 96, 64)
        # We need to resize and potentially adjust the number of channels
        x = F.interpolate(x, size=(96, 64), mode='bilinear', align_corners=False)
        if x.shape[1] == 2:  # If we have 2 channels (mag and phase)
            x = x[:, 0:1, :, :]  # Use only the magnitude
        return self.vggish(x)


def build_feature_extractor():
    return VGGishFeatureExtractor(pretrained=True)