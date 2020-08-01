import torch.nn as nn
from torchvision import models

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19,
            "vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn}


class VGGHash(nn.Module):
    def __init__(self, name, hash_bit):
        super(VGGHash, self).__init__()
        model_vgg = vgg_dict[name](pretrained=True)
        self.features = model_vgg.features

        self.hash_layer = nn.Sequential()
        for i in range(6):
            self.hash_layer.add_module("classifier" + str(i), model_vgg.classifier[i])
        self.hash_layer.add_module("hash", nn.Linear(model_vgg.classifier[6].in_features, hash_bit))

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.hash_layer(x)
        y = self.activation(x)
        return y

    def forward_factor(self, x, factor):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.hash_layer(x)
        y = self.activation(x * factor)
        return y

