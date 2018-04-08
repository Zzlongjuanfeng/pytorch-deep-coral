"""ResNet-34 model for ADDA."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torch.autograd import Variable


class ResNet34Encoder(nn.Module):
    """ResNet-34 encoder model for ADDA."""

    def __init__(self):
        """Init ResNet-34 encoder."""
        super(ResNet34Encoder, self).__init__()

        # input is 224 * 224
        # encoder_list = list(models.resnet34().children())
        # self.encoder = nn.Sequential(*encoder_list[:-1])
        self.encoder = models.resnet34()
        del self.encoder.fc

        self.restored = False

    def forward(self, input):
        """Forward the ResNet-34."""
        x = self.encoder.conv1(input)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        feat = x.view(x.size(0), -1)

        # conv_out = self.encoder(input)
        # feat = conv_out.view(conv_out.size(0), -1)
        # (batch, 512)
        return feat


class Classifier(nn.Module):
    """classifier model for ADDA."""

    def __init__(self):
        """Init classifier."""
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(512, 200)
        self.fc2 = nn.Linear(200, 12)
        self.restored = False

    def forward(self, feat):
        """Forward the classifier."""
        fc1_out = self.fc1(feat)
        out = F.dropout(F.relu(fc1_out), training=self.training)
        out = self.fc2(out)
        return fc1_out, out

if __name__ == '__main__':
    path = '/home/zxf/.torch/models/resnet34-333f7ec4.pth'
    net = ResNet34Encoder()

    weight = torch.load(path)
    weight_net = net.state_dict()
    weight_net_encoder = net.encoder.state_dict()

    net.encoder.load_state_dict(weight, strict=False)

    data = torch.rand([1,3,224,224])
    output = net(Variable(data))